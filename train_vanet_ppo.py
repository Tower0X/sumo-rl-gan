import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction

def train_vanet_agent():
    print("[*] Initialisation de l'environnement VANET Sécurisé...")
    
    # Paramètres de l'environnement
    # On utilise l'environnement simple-intersection défini précédemment
    net_file = 'sumo_rl/nets/2way-single-intersection/single-intersection.net.xml'
    route_file = 'sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml'
    
    # Création de l'environnement avec nos nouvelles métriques
    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name='outputs/vanet_training',
        use_gui=False, # Pas de GUI pour l'entraînement pour aller plus vite
        num_seconds=10000,
        delta_time=5,
        single_agent=True,
        observation_class=VANETObservationFunction, # L'observation avec le jitter Gaussien !
        reward_fn='vanet' # Notre fonction de récompense qui pénalise le freinage d'urgence
    )
    
    print("[*] Environnement créé avec succès. Observation Space:", env.observation_space.shape)
    print("[*] Reward function utilisée: VANET (Fluidité + Pénalités Sécurité/Réseau)")

    # Initialisation de l'agent PPO
    print("[*] Initialisation de l'agent PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./outputs/ppo_vanet_tensorboard/"
    )
    
    # Entraînement de l'agent
    total_timesteps = 20000 # Un petit nombre pour valider que tout fonctionne
    print(f"[*] Début de l'entraînement pour {total_timesteps} étapes (timesteps)...")
    
    try:
        model.learn(total_timesteps=total_timesteps)
        print("[*] Entraînement terminé avec succès !")
        
        # Sauvegarde du modèle
        os.makedirs("outputs", exist_ok=True)
        model.save("outputs/ppo_vanet_model")
        print("[*] Modèle PPO sauvegardé sous 'outputs/ppo_vanet_model.zip'.")
        
    except Exception as e:
        print(f"[!] Une erreur est survenue pendant l'entraînement : {e}")
    finally:
        env.close()

if __name__ == "__main__":
    train_vanet_agent()
