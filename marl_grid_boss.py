import os
import time
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import AttackType, global_orchestrator

def train_urban_shield_4x4():
    print("==================================================================")
    print("🏆 PHASE D'EXCELLENCE : BOUCLIER URBAIN 4x4 (RECURRENT MARL)")
    print("==================================================================")
    print("[*] Architecture : RecurrentPPO (LSTM) + Parameter Sharing")
    print("[*] Réseau : Grille 4x4-Lucas (16 Intersections Intelligentes)")
    print("[*] Intelligence : Capacité de détection des attaques temporelles.\n")

    net_file = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
    route_file = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"

    # 1. Création de l'environnement Parallèle
    print("[*] Initialisation de la Cité Numérique...")
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name="outputs/marl_recurrent_4x4",
        use_gui=False,
        num_seconds=15000, # On augmente le temps de simulation pour laisser le temps au LSTM d'apprendre
        delta_time=5,
        observation_class=VANETObservationFunction,
        reward_fn='vanet'
    )

    # 2. Conversion SuperSuit pour SB3
    print("[*] Conversion Multi-Agent vers Vector-Env (Hive Mind API)...")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)

    # 3. Création du Modèle RecurrentPPO (La Mémoire de la Ville)
    print("[*] Forge de la Matrice Neurale Recurrente (RecurrentPPO)...")
    model = RecurrentPPO(
        "MlpLstmPolicy",  # Politique avec mémoire LSTM intégrée
        env,
        verbose=1,
        learning_rate=2e-4, # LR prudente pour LSTM
        n_steps=512,        # Fenêtre temporelle plus large
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./outputs/recurrent_marl_tensorboard/"
    )

    # 4. Cycle d'Entraînement
    timesteps = 60000
    print(f"\n[*] 🚀 DÉMARRAGE DE LA MUTATION NEURALE ({timesteps} pas)...")
    
    start_time = time.time()
    try:
        model.learn(total_timesteps=timesteps)
        duration = (time.time() - start_time) / 60
        print(f"\n[*] ✅ CONVERGENCE ATTEINTE en {duration:.1f} minutes.")
        
        # Sauvegarde du Modèle d'Excellence
        os.makedirs("outputs", exist_ok=True)
        model.save("outputs/recurrent_urban_shield_4x4")
        print("[*] Le 'Bouclier Urbain' Recurrent est prêt : 'outputs/recurrent_urban_shield_4x4.zip'")
        
    except Exception as e:
        print(f"\n[!] Échec Critique : {e}")
    finally:
        env.close()

if __name__ == "__main__":
    train_urban_shield_4x4()
