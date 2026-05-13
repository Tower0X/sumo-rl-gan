import time
import gymnasium as gym
from stable_baselines3 import PPO

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import global_orchestrator, AttackType

def run_manual_override_demo():
    print("======================================================")
    print("🎮 CONTRÔLEUR D'ATTAQUE : DÉMO MANUAL OVERRIDE")
    print("======================================================\n")

    env = sumo_rl.SumoEnvironment(
        net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
        route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
        use_gui=True, # Interface graphique pour voir les dégâts
        num_seconds=1000, 
        delta_time=5,
        single_agent=True,
        observation_class=VANETObservationFunction,
        reward_fn='vanet'
    )

    try:
        model = PPO.load("outputs/ppo_vanet_model")
    except FileNotFoundError:
        print("[!] Modèle introuvable. Avez-vous lancé train_vanet_ppo.py ?")
        return

    obs, info = env.reset()
    done = False
    step = 0
    ts_id = env.ts_ids[0]

    while not done and step < 120:
        # L'IA prend sa décision basée sur ce que l'Orchestrateur lui laisse voir
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- SÉQUENCE D'ATTAQUES MANUELLES PROGRAMMÉES ---
        
        if step == 20:
            print("\n[🗡️ ATTAQUE 1] Injection de Véhicules Fantômes (Sybil Attack)")
            print("-> L'IA va croire que les routes sont saturées et risque de paniquer.")
            global_orchestrator.trigger_manual_attack(ts_id, AttackType.GHOST_VEHICLES, intensity=0.9, duration_steps=10)
            
        elif step == 50:
            print("\n[🗡️ ATTAQUE 2] Empoisonnement des Données (Data Poisoning)")
            print("-> On cache les embouteillages réels à l'IA (Déni de réalité).")
            global_orchestrator.trigger_manual_attack(ts_id, AttackType.DATA_POISONING, intensity=0.9, duration_steps=10)

        elif step == 80:
            print("\n[🗡️ ATTAQUE 3] Temporal DoS Massif (Coupure Totale V2X)")
            print("-> Brouillage du signal et coupure RSU. L'IA doit passer en mode Fail-Safe.")
            global_orchestrator.trigger_manual_attack(ts_id, AttackType.TEMPORAL_DOS, intensity=1.0, duration_steps=15)

        # Affichage
        if step % 5 == 0:
            etat = "🟢 SAIN" if ts_id not in global_orchestrator.active_attacks else "🔴 SOUS ATTAQUE"
            print(f"Pas {step:03d} | Réseau : {etat} | Récompense : {reward:+.2f}")
            time.sleep(0.2)

        done = terminated or truncated
        step += 1

    env.close()

if __name__ == "__main__":
    run_manual_override_demo()
