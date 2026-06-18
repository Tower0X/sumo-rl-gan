"""Vrai MARL cooperatif : defenseur RecurrentPPO sur la grille urbaine 4x4.

Architecture (classe mondiale, defendable) :
  - Environnement multi-agents PettingZoo (``sumo_rl.parallel_env``) : les 16
    feux de la grille 4x4 sont des agents simultanes.
  - Parameter sharing via SuperSuit : un SEUL cerveau partage par les 16 feux
    (apprentissage 16x plus dense, resilience distribuee, generalisation).
  - RecurrentPPO (LSTM) : le cerveau possede une MEMOIRE temporelle, coherent
    avec le README et indispensable face a des attaques sequentielles (DoS
    persistant, Sybil graduel) ou l'historique recent est discriminant.
  - Observation VANET (latence V2X + comm_flag) et recompense Fail-Safe
    2-phases (``reward_fn='vanet'``) conservees telles quelles.

Sortie : ``outputs/recurrent_urban_shield_4x4.zip`` (le nom attendu par
``sim_runner.py`` et ``evaluate_gan_vs_defender.py``).

Prerequis :
  - SUMO_HOME configure
  - pip install sb3-contrib supersuit stable-baselines3

Usage :
  python train_marl_defender.py --timesteps 200000
"""
import argparse
import os

import supersuit as ss
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecMonitor

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction


NET_FILE = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
ROUTE_FILE = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"
MODEL_PATH = "outputs/recurrent_urban_shield_4x4"
TENSORBOARD_DIR = "outputs/marl_tensorboard"


def build_marl_env(num_seconds=10000, use_gui=False, out_csv="outputs/marl_4x4_training"):
    """Construit l'environnement MARL vectorise pour le parameter sharing.

    Renvoie un VecEnv compatible Stable-Baselines3 ou les 16 agents-feux
    sont concatenes et controles par un cerveau partage.
    """
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=5,
        observation_class=VANETObservationFunction,
        reward_fn="vanet",
        collision_action="warn",   # collisions reellement mesurees
        time_to_teleport=300,       # collisions physiquement possibles (pas masquees)
    )

    # SuperSuit : PettingZoo -> VecEnv SB3 (parameter sharing)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def train(timesteps=200000, num_seconds=10000, use_gui=False):
    print("=" * 66)
    print("DEFENSEUR MARL COOPERATIF : GRILLE URBAINE 4x4 (16 AGENTS)")
    print("=" * 66)
    print("[*] Methode : RecurrentPPO (LSTM) + Parameter Sharing (SuperSuit)")
    print("[*] Les 16 feux partagent un cerveau temporel unique.\n")

    os.makedirs("outputs", exist_ok=True)
    env = build_marl_env(num_seconds=num_seconds, use_gui=use_gui)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=TENSORBOARD_DIR,
    )

    print(f"[*] Demarrage de l'entrainement MARL ({timesteps} etapes)...")
    try:
        model.learn(total_timesteps=timesteps, tb_log_name="marl_recurrent_defender")
        model.save(MODEL_PATH)
        print(f"\n[OK] Cerveau MARL recurrent sauvegarde -> {MODEL_PATH}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrainement du defenseur MARL recurrent (grille 4x4).")
    parser.add_argument("--timesteps", type=int, default=200000, help="Nombre total d'etapes d'entrainement.")
    parser.add_argument("--num-seconds", type=int, default=10000, help="Duree (s) d'un episode SUMO.")
    parser.add_argument("--gui", action="store_true", help="Afficher SUMO-GUI pendant l'entrainement.")
    args = parser.parse_args()
    train(timesteps=args.timesteps, num_seconds=args.num_seconds, use_gui=args.gui)
