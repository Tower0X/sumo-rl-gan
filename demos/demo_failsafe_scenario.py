import time
import os
import gymnasium as gym
from stable_baselines3 import PPO

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction

def run_attack_scenario():
    print("======================================================")
    print("[*] DÉMARRAGE DU SCÉNARIO VANET : RÉSILIENCE AUX CYBERATTAQUES")
    print("======================================================\n")

    net_file = 'sumo_rl/nets/2way-single-intersection/single-intersection.net.xml'
    route_file = 'sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml'

    print("[*] Initialisation de la ville virtuelle et de l'interface graphique...")
    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=True, # Interface graphique activée
        num_seconds=1000, 
        delta_time=5,
        single_agent=True,
        observation_class=VANETObservationFunction,
        reward_fn='vanet'
    )

    print("[*] Chargement du 'cerveau' PPO entraîné...")
    try:
        model = PPO.load("outputs/ppo_vanet_model")
    except FileNotFoundError:
        print("[!] Erreur: Le modèle PPO n'a pas été trouvé. Assurez-vous que l'entraînement s'est bien terminé.")
        return

    obs, info = env.reset()
    done = False
    step = 0
    attack_triggered = False
    ts_id = env.ts_ids[0]

    print("\n[*] 🟢 PHASE 1 : Trafic Normal (Réseau V2X Sain)")
    print("[*] L'agent PPO optimise le trafic grâce aux données V2X.")

    # On fait tourner la simulation pour un nombre limité d'étapes (ex: 80 pas = 400s)
    while not done and step < 80:
        # 1. L'IA analyse son environnement et prend une décision
        action, _states = model.predict(obs, deterministic=True)
        
        # 2. On applique la décision à l'environnement SUMO
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. DÉCLENCHEMENT DE L'ATTAQUE (À t = 40)
        if step == 40 and not attack_triggered:
            print("\n======================================================")
            print("[!!!] ALERTE CRITIQUE : DÉTECTION DE BROUILLAGE [!!!]")
            print("[!!!] TYPE D'ATTAQUE : Déni de Service Géolocalisé (Geo-DoS)")
            print("[!!!] CIBLE : Infrastructure RSU Principale")
            print("======================================================")
            print("[*] 🔴 PHASE 2 : Réseau Hors Ligne - Passage en mode 'Fail-Safe'\n")
            
            # Injection de la panne dans notre environnement !
            env.traffic_signals[ts_id].comm_ok = False
            attack_triggered = True

        # 4. Affichage de la Télémétrie
        if step % 5 == 0 or (attack_triggered and step % 2 == 0):
            status = "🔴 HORS LIGNE (BROUILLÉ)" if attack_triggered else "🟢 EN LIGNE (OPTIMAL)"
            print(f"-> Pas {step:02d} | Réseau : {status} | Récompense : {reward:+.2f} | File d'attente estimée : {info.get(f'{ts_id}_stopped', 0)}")
            
            if attack_triggered:
                # Petit délai pour laisser le temps d'observer le comportement sous attaque à l'écran
                time.sleep(0.3) 

        done = terminated or truncated
        step += 1

    env.close()
    print("\n======================================================")
    print("[*] FIN DU SCÉNARIO D'ATTAQUE")
    print("[*] Observez comment l'agent a évité la catastrophe malgré la perte des communications.")
    print("======================================================")

if __name__ == "__main__":
    run_attack_scenario()
