"""Moteur de simulation MARL pour le dashboard de supervision VANET.

Vrai MARL : environnement PettingZoo parallele (16 feux) controle par un
cerveau RecurrentPPO partage (parameter sharing). Les attaques (manuelles ou
surrogate) ciblent les intersections individuellement. Les metriques exposees
au dashboard sont agregees sur les 16 agents.
"""
import time

import numpy as np
import torch

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import (
    global_orchestrator,
    AttackType,
    compute_obs_layout,
)
from sumo_rl.environment.gan_attacker import load_generator_strict, GANLoadError
from sb3_contrib import RecurrentPPO
from shared_state import state


NET_FILE = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
ROUTE_FILE = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"
RECURRENT_PATH = "outputs/recurrent_urban_shield_4x4"


def run_simulation():
    state.reset()
    state.running = True
    state.add_log("Demarrage du bouclier urbain MARL (16 agents)...", "info")

    # 1. Environnement MARL parallele (16 feux simultanes)
    try:
        par_env = sumo_rl.parallel_env(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            use_gui=state.use_gui,
            num_seconds=20000,
            delta_time=5,
            observation_class=VANETObservationFunction,
            reward_fn="vanet",
            collision_action="warn",
            time_to_teleport=300,
        )
    except Exception as e:
        state.add_log(f"Erreur SUMO: {str(e)}", "error")
        state.running = False
        return

    ts_ids = list(par_env.possible_agents)
    with state.lock:
        state.available_nodes = ts_ids
        if not state.target_node_id and ts_ids:
            state.target_node_id = ts_ids[0]

    # 2. Defenseur MARL recurrent (cerveau partage)
    state.add_log("Chargement de la matrice neurale recurrente (LSTM partagee)...", "info")
    try:
        defender = RecurrentPPO.load(RECURRENT_PATH)
    except Exception:
        state.add_log("Modele MARL recurrent introuvable. Entrainez train_marl_defender.py.", "error")
        par_env.close()
        state.running = False
        return

    # 3. Attaquant surrogate (STRICT: jamais de poids aleatoires)
    state.add_log("Chargement de l'attaquant surrogate (LSTM)...", "info")
    sample_agent = par_env.possible_agents[0]
    state_dim = par_env.observation_space(sample_agent).shape[0]
    gan_loaded = False
    attacker = None
    attacker_hidden = None
    try:
        attacker = load_generator_strict(state_dim)
        gan_loaded = True
    except GANLoadError as exc:
        state.add_log(f"Attaquant non charge ({exc}). Mode adversarial desactive.", "warning")

    observations, infos = par_env.reset()
    lstm_states = {agent: None for agent in par_env.possible_agents}
    episode_starts = {agent: True for agent in par_env.possible_agents}

    # Acces a l'env SUMO sous-jacent pour piloter comm_ok / layout.
    sumo_env = par_env.unwrapped.env if hasattr(par_env, "unwrapped") else None

    state.add_log("Supervision urbaine (16 agents) active.", "info")

    while par_env.agents and not state.should_stop:
        with state.lock:
            current_mode = state.mode
            manual_atk = state.manual_attack_type
            manual_virulence = state.manual_virulence
            target_ts = state.target_node_id

        active_name = "Aucune"
        active_intensity = 0.0

        # --- GESTION DES ATTAQUES ---
        if current_mode == "defense_only":
            global_orchestrator.active_attacks.clear()
            if sumo_env is not None:
                for tid in ts_ids:
                    if tid in sumo_env.traffic_signals:
                        sumo_env.traffic_signals[tid].comm_ok = True

        elif current_mode == "manual_attack":
            if manual_atk != AttackType.NONE and target_ts:
                global_orchestrator.trigger_manual_attack(
                    target_ts, manual_atk, manual_virulence, duration_steps=1
                )
                active_name = f"{manual_atk.name} sur {target_ts}"
                active_intensity = manual_virulence

        elif current_mode == "adversarial_gan":
            if not gan_loaded:
                state.add_log("Mode surrogate demande mais aucun attaquant charge. Ignore.", "error")
            elif target_ts in observations:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(observations[target_ts]).unsqueeze(0)
                    attack_vec_t, attacker_hidden = attacker(obs_t, attacker_hidden)
                    attack_vec = attack_vec_t.numpy()[0]
                global_orchestrator.bridge_cGAN_tensor(target_ts, attack_vec)
                if target_ts in global_orchestrator.active_attacks:
                    atk = global_orchestrator.active_attacks[target_ts]
                    active_name = f"SURROGATE: {atk['type'].name}"
                    active_intensity = atk["intensity"]

        # --- DECISION DU DEFENSEUR (par agent, cerveau partage) ---
        actions = {}
        for agent, obs in observations.items():
            action, lstm_states[agent] = defender.predict(
                obs,
                state=lstm_states[agent],
                episode_start=np.array([episode_starts[agent]]),
                deterministic=True,
            )
            episode_starts[agent] = False
            actions[agent] = int(action)

        observations, rewards, terminations, truncations, infos = par_env.step(actions)

        # --- AGREGATION DES METRIQUES SUR LES 16 AGENTS ---
        with state.lock:
            state.step += 1
            state.active_attack_name = active_name
            state.active_attack_intensity = active_intensity

            mean_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            state.current_reward = mean_reward
            state.reward_history.append(mean_reward)

            # Temps d'attente systeme (depuis info systeme commun a tous les agents)
            sys_wait = 0.0
            for ag_info in infos.values():
                sys_wait = ag_info.get("system_total_waiting_time", sys_wait)
                break
            state.waiting_time_history.append(sys_wait)

            # Latence et files agregees via indexation EXACTE (compute_obs_layout)
            total_latency = 0.0
            total_queues = 0
            n = 0
            for agent, obs in observations.items():
                ts = sumo_env.traffic_signals.get(agent) if sumo_env is not None else None
                if ts is None:
                    continue
                layout = compute_obs_layout(ts)
                oq = layout["offset_queue"]
                n_lanes = layout["n_lanes"]
                total_latency += float(np.abs(obs[layout["latency_idx"]]) * 10)
                total_queues += int(np.sum(obs[oq:oq + n_lanes]) * 10)
                n += 1
            state.latency_history.append(total_latency / n if n else 0.0)
            state.queue_history.append(total_queues)

        if state.use_gui:
            time.sleep(0.02)

    par_env.close()
    with state.lock:
        state.running = False
    state.add_log("Simulation arretee.", "info")
