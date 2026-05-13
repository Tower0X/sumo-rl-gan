import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction

def train_marl_cooperative_defense():
    print("==================================================================")
    print("🛡️ DÉPLOIEMENT MARL AUDACIEUX : GRILLE URBAINE 4x4 (16 AGENTS)")
    print("==================================================================")
    print("[*] Objectif : Préparer l'architecture de défense coopérative pour le GAN.")
    print("[*] Méthode : Parameter Sharing (Les 16 feux partagent un seul et même 'Cerveau' PPO).")
    print("[*] Avantage : Apprentissage 16x plus rapide. Résilience distribuée.\n")

    # Utilisation du réseau 4x4 existant
    net_file = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
    route_file = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"

    # Création de l'environnement Multi-Agent PettingZoo (Parallel API)
    print("[*] Initialisation de l'environnement parallèle PettingZoo...")
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name="outputs/marl_4x4_training",
        use_gui=False,
        num_seconds=10000,
        delta_time=5,
        observation_class=VANETObservationFunction,
        reward_fn='vanet' # On garde notre fonction sécurisée !
    )

    # ---------------------------------------------------------
    # LA TOUCHE D'EXCELLENCE : SUPERSUIT WRAPPERS
    # ---------------------------------------------------------
    # Pour que Stable Baselines 3 puisse entraîner 16 agents simultanément
    # dans une même simulation, nous convertissons l'environnement PettingZoo
    # en un VectorEnvironment classique grâce à SuperSuit.
    # Cela permet le "Parameter Sharing" : le PPO apprend des expériences 
    # de TOUS les carrefours en même temps. S'il résout une attaque au Nord, 
    # le carrefour au Sud saura immédiatement s'en défendre !
    
    print("[*] Application du blindage (SuperSuit) pour le Parameter Sharing...")
    try:
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
        env = VecMonitor(env)
    except Exception as e:
        print(f"[!] Erreur de configuration SuperSuit : {e}")
        print("[!] Veuillez installer supersuit : pip install supersuit")
        return

    # Initialisation de l'Agent PPO Partagé
    print("[*] Création de la Matrice PPO Centrale...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./outputs/marl_tensorboard/"
    )

    # Entraînement
    total_timesteps = 50000 # Entraînement plus long justifié par la complexité 4x4
    print(f"\n[*] 🚀 DÉMARRAGE DE L'ENTRAÎNEMENT MARL ({total_timesteps} étapes)...")
    
    try:
        model.learn(total_timesteps=total_timesteps)
        print("\n[*] ✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
        
        # Sauvegarde
        os.makedirs("outputs", exist_ok=True)
        model.save("outputs/ppo_marl_4x4_model")
        print("[*] Le Cerveau Centralisé (Hive Mind) a été sauvegardé sous 'outputs/ppo_marl_4x4_model.zip'.")
        
    except Exception as e:
        print(f"\n[!] Erreur détectée lors de l'entraînement : {e}")
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    train_marl_cooperative_defense()
