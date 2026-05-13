import time
import torch
import gymnasium as gym
from stable_baselines3 import PPO

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import global_orchestrator, AttackType
from sumo_rl.environment.gan_attacker import Generator

def run_final_battle():
    print("======================================================")
    print("🏁 LA BATAILLE FINALE : IA DÉFENSE vs IA ATTAQUE (GAN)")
    print("======================================================\n")

    # 1. Init Environnement avec GUI
    env = sumo_rl.SumoEnvironment(
        net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
        route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
        use_gui=True,
        num_seconds=2000, 
        delta_time=5,
        single_agent=True,
        observation_class=VANETObservationFunction,
        reward_fn='vanet'
    )

    # 2. Chargement du Défenseur (PPO)
    print("[*] Chargement du Défenseur PPO...")
    try:
        defender = PPO.load("outputs/ppo_vanet_model")
    except:
        print("[!] Erreur: ppo_vanet_model introuvable.")
        return

    # 3. Chargement de l'Attaquant (GAN)
    print("[*] Chargement de l'Attaquant GAN (Neural Hacker)...")
    state_dim = env.observation_space.shape[0]
    gan_hacker = Generator(state_dim)
    try:
        gan_hacker.load_state_dict(torch.load("outputs/gan/generator_model.pth"))
        gan_hacker.eval()
    except:
        print("[!] Erreur: generator_model.pth introuvable.")
        return

    obs, info = env.reset()
    done = False
    step = 0
    ts_id = env.ts_ids[0]

    print("\n🚀 DÉBUT DU DUEL EN DIRECT...")
    while not done:
        # --- TOUR DU GAN (ATTAQUE) ---
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            attack_vector = gan_hacker(state_tensor).numpy()[0]
            
        # Exécution de l'attaque générée par l'IA
        global_orchestrator.bridge_cGAN_tensor(ts_id, attack_vector)
        
        # --- TOUR DU PPO (DÉFENSE) ---
        action, _ = defender.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Affichage du statut
        if step % 2 == 0:
            attack_info = "🟢 AUCUNE"
            if ts_id in global_orchestrator.active_attacks:
                atk = global_orchestrator.active_attacks[ts_id]
                attack_info = f"🔴 {atk['type'].name} (Force: {atk['intensity']*100:.1f}%)"
            
            print(f"Pas {step:03d} | Attaque GAN : {attack_info} | Récompense : {reward:+.2f}")
            time.sleep(0.1) # Ralentir un peu pour que ce soit visible dans la GUI

        done = terminated or truncated
        step += 1

    env.close()
    print("\n[🏁] Duel terminé.")

if __name__ == "__main__":
    run_final_battle()
