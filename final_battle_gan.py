import time
import torch
import numpy as np
from sb3_contrib import RecurrentPPO
import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import global_orchestrator, AttackType
from sumo_rl.environment.gan_attacker import load_generator_strict, GANLoadError

def run_excellence_battle():
    print("======================================================")
    print("🏆 DUEL D'EXCELLENCE : BOUCLIER URBAIN vs LSTM-GAN")
    print("======================================================\n")

    # 1. Init Environnement 4x4 avec GUI
    print("[*] Initialisation de la Grille Urbaine (4x4)...")
    env = sumo_rl.SumoEnvironment(
        net_file='sumo_rl/nets/4x4-Lucas/4x4.net.xml',
        route_file='sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
        use_gui=True,
        num_seconds=4000, 
        delta_time=5,
        single_agent=True, # Utilisation du modèle partagé sur l'espace d'action global
        observation_class=VANETObservationFunction,
        reward_fn='vanet',
        collision_action='warn'  # Collisions réellement mesurées
    )

    # 2. Chargement du Défenseur RecurrentPPO (LSTM)
    print("[*] Chargement du Défenseur Temporel (RecurrentPPO)...")
    try:
        defender = RecurrentPPO.load("outputs/recurrent_urban_shield_4x4")
    except:
        print("[!] Erreur: recurrent_urban_shield_4x4 introuvable. Tentative avec le modèle MLP...")
        try:
            from stable_baselines3 import PPO
            defender = PPO.load("outputs/ppo_marl_4x4_model")
        except:
            print("[!] Échec critique: aucun modèle de défense trouvé.")
            return

    # 3. Chargement de l'Attaquant LSTM-GAN (STRICT: pas de duel sur poids aléatoires)
    print("[*] Chargement du Hacker Temporel (LSTM-GAN)...")
    state_dim = env.observation_space.shape[0]
    try:
        gan_hacker = load_generator_strict(state_dim)
    except GANLoadError as exc:
        print(f"[!] Échec critique du chargement du GAN: {exc}")
        print("[!] Duel annulé: refus de combattre un attaquant non entraîné (poids aléatoires).")
        env.close()
        return

    obs, info = env.reset()
    done = False
    step = 0
    
    # Mémoire du GAN (états cachés LSTM)
    gan_hidden = None

    print("\n🚀 DÉBUT DE LA SUPERVISION URBAINE...")
    while not done:
        # --- TOUR DU GAN (ATTAQUE SÉQUENTIELLE) ---
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0) # (Batch=1, Seq=1, Dim)
            attack_vector_tensor, gan_hidden = gan_hacker(state_tensor, gan_hidden)
            attack_vector = attack_vector_tensor.numpy()[0]
            
        # Ciblage d'une intersection aléatoire ou critique pour la démo
        target_ts = env.ts_ids[step % len(env.ts_ids)]
        global_orchestrator.bridge_cGAN_tensor(target_ts, attack_vector)
        
        # --- TOUR DU PPO (DÉFENSE AVEC MÉMOIRE) ---
        # RecurrentPPO gère son propre état interne si on ne lui passe pas
        action, _ = defender.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Affichage du statut toutes les 10 étapes
        if step % 10 == 0:
            attack_info = "🟢 SAIN"
            if target_ts in global_orchestrator.active_attacks:
                atk = global_orchestrator.active_attacks[target_ts]
                attack_info = f"🔴 {atk['type'].name} sur {target_ts}"
            
            print(f"Step {step:03d} | État Ville : {attack_info} | Récompense : {reward:+.2f}")

        done = terminated or truncated
        step += 1
        time.sleep(0.05)

    env.close()
    print("\n[🏁] Mission terminée.")

if __name__ == "__main__":
    run_excellence_battle()
