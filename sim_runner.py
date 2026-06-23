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


def run_simulation():
    state.reset()
    state.running = True
    state.add_log("Demarrage du bouclier urbain MARL...", "info")

    # Sélection de la ville
    if state.city_map == "Ville A (Grille 2x2)":
        net_file = "sumo_rl/nets/2x2/2x2.net.xml"
        route_file = "sumo_rl/nets/2x2/2x2.rou.xml"
    elif state.city_map == "Ville B (Grille 3x2)":
        net_file = "sumo_rl/nets/3x2/3x2.net.xml"
        route_file = "sumo_rl/nets/3x2/3x2.rou.xml"
    else:
        net_file = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
        route_file = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"

    # 1. Environnement MARL parallele
    try:
        par_env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=route_file,
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
        defender = RecurrentPPO.load("outputs/recurrent_urban_shield_4x4")
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
                # Durée généreuse + ré-armement chaque step: l'attaque survit
                # continuellement tant que l'utilisateur la maintient active.
                global_orchestrator.trigger_manual_attack(
                    target_ts, manual_atk, manual_virulence, duration_steps=20
                )
                active_name = f"{manual_atk.name} sur {target_ts}"
                active_intensity = manual_virulence
                # Log visible au dashboard (pas à chaque step pour éviter le spam)
                if state.step % 5 == 0:
                    state.add_log(
                        f"ATTAQUE {manual_atk.name} (force {manual_virulence*100:.0f}%) "
                        f"sur {target_ts}", "warning"
                    )

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

        # --- CANAL PHYSIQUE: effet TraCI VISIBLE sur TOUTES les intersections attaquées ---
        # Après le step, SUMO est avancé: on injecte les véhicules fantômes
        # rouges / on gèle la phase pour rendre l'attaque physiquement réelle.
        physical_effects = {"ghosts_spawned": 0, "phase_frozen": False}
        if sumo_env is not None and current_mode in ("manual_attack", "adversarial_gan"):
            for atk_ts_id in list(global_orchestrator.active_attacks.keys()):
                if atk_ts_id in sumo_env.traffic_signals:
                    fx = global_orchestrator.apply_physical_attack(
                        sumo_env.traffic_signals[atk_ts_id], sumo_env.sumo
                    )
                    physical_effects["ghosts_spawned"] += fx.get("ghosts_spawned", 0)
                    physical_effects["phase_frozen"] = physical_effects["phase_frozen"] or fx.get("phase_frozen", False)

        # --- AGREGATION DES METRIQUES SUR LES 16 AGENTS ---
        # Track comm_ok de la cible pour le dashboard
        target_comm_ok = True
        if sumo_env is not None and target_ts in sumo_env.traffic_signals:
            target_comm_ok = getattr(sumo_env.traffic_signals[target_ts], "comm_ok", True)

        with state.lock:
            state.step += 1
            state.active_attack_name = active_name
            state.active_attack_intensity = active_intensity
            state.total_ghosts_spawned += physical_effects.get("ghosts_spawned", 0)
            state.phase_frozen = physical_effects.get("phase_frozen", False)
            state.target_comm_ok = target_comm_ok

            # Nombre d'intersections actuellement sous attaque
            state.n_intersections_attacked = len(global_orchestrator.active_attacks)

            mean_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            state.current_reward = mean_reward
            state.reward_history.append(mean_reward)

            # Reward de la cible seule (pour voir l'impact direct)
            if target_ts in rewards:
                state.target_reward = float(rewards[target_ts])
            state.target_reward_history.append(state.target_reward)

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
                ts_obj = sumo_env.traffic_signals.get(agent) if sumo_env is not None else None
                if ts_obj is None:
                    continue
                layout = compute_obs_layout(ts_obj)
                oq = layout["offset_queue"]
                n_lanes = layout["n_lanes"]
                # Latence normalisée [0,1] -> affichée en ms (×100)
                total_latency += float(np.abs(obs[layout["latency_idx"]]) * 100)
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
