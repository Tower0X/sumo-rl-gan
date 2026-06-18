"""Evaluation du duel : DEFENSEUR MARL (16 feux) vs ATTAQUANT SURROGATE (LSTM).

Vrai MARL : l'environnement PettingZoo parallele expose les 16 feux de la
grille 4x4 comme agents simultanes ; le cerveau RecurrentPPO partage les
controle tous (parameter sharing). L'attaquant surrogate genere des attaques
adaptatives ciblant chaque intersection par son id, via l'orchestrateur
cyber-physique.

Chargement STRICT de l'attaquant : aucun duel sur poids aleatoires.
"""
import os
import time

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import global_orchestrator
from sumo_rl.environment.gan_attacker import load_generator_strict, GANLoadError


NET_FILE = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
ROUTE_FILE = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"
RECURRENT_PATH = "outputs/recurrent_urban_shield_4x4"


def evaluate_gan_vs_defender(num_seconds=4000, use_gui=True):
    print("======================================================")
    print("EVALUATION : DEFENSEUR MARL (16 feux) vs ATTAQUANT SURROGATE (LSTM)")
    print("======================================================\n")

    # 1. Environnement MARL parallele (16 agents simultanes)
    print("[*] Initialisation de la grille urbaine 4x4 (MARL, 16 agents)...")
    par_env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=5,
        observation_class=VANETObservationFunction,
        reward_fn="vanet",
        collision_action="warn",
        time_to_teleport=300,
    )

    # 2. Defenseur RecurrentPPO partage
    print("[*] Chargement du defenseur MARL recurrent (cerveau partage)...")
    if not os.path.exists(RECURRENT_PATH + ".zip"):
        print(f"[!] Modele de defense introuvable: {RECURRENT_PATH}.zip")
        print("[!] Entrainez d'abord le defenseur: python train_marl_defender.py")
        par_env.close()
        return
    defender = RecurrentPPO.load(RECURRENT_PATH)

    # 3. Attaquant surrogate (chargement STRICT)
    print("[*] Chargement de l'attaquant surrogate (LSTM)...")
    sample_agent = par_env.possible_agents[0]
    state_dim = par_env.observation_space(sample_agent).shape[0]
    try:
        attacker = load_generator_strict(state_dim)
    except GANLoadError as exc:
        print(f"[!] Echec critique du chargement de l'attaquant: {exc}")
        print("[!] Duel annule: refus de combattre un attaquant non entraine.")
        par_env.close()
        return

    observations, infos = par_env.reset()
    # Etats LSTM du defenseur, un par agent (parameter sharing).
    lstm_states = {agent: None for agent in par_env.possible_agents}
    episode_starts = {agent: True for agent in par_env.possible_agents}
    attacker_hidden = None

    print("\nDEBUT DE LA SUPERVISION URBAINE...")
    step = 0
    while par_env.agents:
        # --- TOUR DE L'ATTAQUANT (sequence adaptative) ---
        target_ts = par_env.agents[step % len(par_env.agents)]
        if target_ts in observations:
            with torch.no_grad():
                obs_t = torch.FloatTensor(observations[target_ts]).unsqueeze(0)
                attack_vec_t, attacker_hidden = attacker(obs_t, attacker_hidden)
                attack_vec = attack_vec_t.numpy()[0]
            global_orchestrator.bridge_cGAN_tensor(target_ts, attack_vec)

        # --- TOUR DU DEFENSEUR (decision par agent, cerveau partage) ---
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

        if step % 10 == 0:
            attack_info = "SAIN"
            if target_ts in global_orchestrator.active_attacks:
                atk = global_orchestrator.active_attacks[target_ts]
                attack_info = f"ATTAQUE {atk['type'].name} sur {target_ts}"
            mean_r = float(np.mean(list(rewards.values()))) if rewards else 0.0
            print(f"Step {step:03d} | Etat ville : {attack_info} | Reward moy : {mean_r:+.2f}")

        step += 1
        if use_gui:
            time.sleep(0.02)

    par_env.close()
    print("\n[FIN] Mission terminee.")


if __name__ == "__main__":
    evaluate_gan_vs_defender()
