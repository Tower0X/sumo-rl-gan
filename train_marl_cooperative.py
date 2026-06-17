"""Cooperative multi-agent training for the 4x4 VANET grid (parameter sharing).

The 4x4-Lucas network has 16 traffic lights. Instead of controlling a single
junction with ``single_agent=True``, we use the PettingZoo ``parallel_env``
together with SuperSuit so that ONE shared policy is trained from the pooled
experience of ALL 16 junctions (parameter sharing).

The policy is a RecurrentPPO (LSTM), which matches the temporal defender loaded
by the dashboard (``sim_runner.py``) and the evaluation script
(``evaluate_gan_vs_defender.py``) from ``outputs/recurrent_urban_shield_4x4``.
"""

import os
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction


NET_FILE = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
ROUTE_FILE = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"
MODEL_PATH = "outputs/recurrent_urban_shield_4x4"


def train_marl_cooperative(total_timesteps: int = 60000, num_seconds: int = 15000):
    """Train a shared RecurrentPPO defender over the 16-junction grid.

    Args:
        total_timesteps: number of environment steps to train for.
        num_seconds: simulated seconds per SUMO episode.
    """
    print("==================================================================")
    print("  Cooperative MARL training - 4x4 grid (16 traffic lights)")
    print("  Shared RecurrentPPO policy via PettingZoo + SuperSuit")
    print("==================================================================\n")

    # 1. Multi-agent environment: one agent per traffic light.
    print("[*] Building PettingZoo parallel environment...")
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name="outputs/marl_recurrent_4x4",
        use_gui=False,
        num_seconds=num_seconds,
        delta_time=5,
        observation_class=VANETObservationFunction,
        reward_fn="vanet",
        collision_action="warn",  # train against REAL collisions
    )

    # 2. Parameter sharing: convert the multi-agent env into a single vec env
    #    so one policy learns from every junction simultaneously.
    print("[*] Wrapping with SuperSuit for parameter sharing...")
    try:
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
        env = VecMonitor(env)
    except Exception as exc:
        print(f"[!] SuperSuit wrapping failed: {exc}")
        print("[!] Install SuperSuit with: pip install supersuit")
        return

    # 3. Recurrent (LSTM) defender, matching the dashboard/evaluator expectation.
    print("[*] Creating RecurrentPPO (MlpLstmPolicy) defender...")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./outputs/recurrent_marl_tensorboard/",
    )

    print(f"\n[*] Training for {total_timesteps} timesteps...")
    start = time.time()
    try:
        model.learn(total_timesteps=total_timesteps)
        minutes = (time.time() - start) / 60
        print(f"\n[*] Training finished in {minutes:.1f} min.")
        os.makedirs("outputs", exist_ok=True)
        model.save(MODEL_PATH)
        print(f"[*] Shared defender saved to '{MODEL_PATH}.zip'.")
    except Exception as exc:
        print(f"\n[!] Training failed: {exc}")
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    train_marl_cooperative()
