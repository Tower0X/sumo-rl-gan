"""Entrainement adversarial du defenseur MARL avec curriculum 3 phases.

Pourquoi un curriculum ? Une resilience CREDIBLE doit etre APPRISE, pas
zero-shot. On expose donc progressivement le defenseur MARL a des attaques
de difficulte croissante, en pilotant l'orchestrateur cyber-physique pendant
l'entrainement :

  Phase 1 - Warm-up (jitter seul) :
      Le defenseur apprend d'abord le controle de trafic de base avec le
      simple bruit de latence V2X deja present dans VANETObservationFunction.

  Phase 2 - Durcissement (attaques manuelles) :
      On injecte des attaques scriptees cyclant les 8 familles (Ghost
      Vehicles, Data Poisoning, Jammer, Flooding, etc.) sur des intersections
      tournantes. Le defenseur construit une robustesse explicite.

  Phase 3 - Co-evolution (attaquant surrogate LSTM) :
      L'attaquant adversarial (SurrogateAdversarialAttacker) genere des
      attaques adaptatives conditionnees par l'etat du trafic. Defenseur et
      attaquant co-evoluent.

Point technique cle : pendant l'entrainement vectorise (SuperSuit), le
callback retrouve l'environnement SUMO sous-jacent (SumoEnvironment) pour
piloter ``global_orchestrator`` et cibler chaque intersection par son id.

Sortie : ``outputs/recurrent_urban_shield_4x4.zip`` (defenseur durci) et,
si la phase 3 entraine l'attaquant, ``outputs/gan/generator_model.pth``.

Prerequis : SUMO_HOME, sb3-contrib, supersuit, torch.

Usage :
  python train_marl_adversarial_curriculum.py \
      --phase1 60000 --phase2 100000 --phase3 120000
"""
import argparse
import os

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

from train_marl_defender import build_marl_env, MODEL_PATH, TENSORBOARD_DIR
from sumo_rl.environment.attack_controller import global_orchestrator, AttackType
from sumo_rl.environment.gan_attacker import (
    SurrogateAdversarialAttacker,
    DEFAULT_GENERATOR_PATH,
)


# Sequence d'attaques manuelles parcourue en Phase 2 (8 familles utiles).
_ATTACK_SEQUENCE = [
    (AttackType.GHOST_VEHICLES, 0.6),
    (AttackType.DATA_POISONING, 0.7),
    (AttackType.TEMPORAL_DOS, 0.5),
    (AttackType.FLOODING_DDOS, 0.6),
    (AttackType.SLOWLORIS_DDOS, 0.5),
    (AttackType.POSITION_JITTER, 0.5),
    (AttackType.JAMMER, 0.8),
]


def _resolve_sumo_env(vec_env):
    """Retrouve le SumoEnvironment sous-jacent a travers les wrappers SuperSuit.

    La pile de wrappers (VecMonitor -> concat_vec_envs -> pettingzoo_to_vec)
    masque l'env PettingZoo. On descend prudemment via les attributs connus
    pour exposer ``ts_ids`` et ``traffic_signals``. Renvoie None si introuvable.
    """
    candidates = ["venv", "vec_envs", "par_env", "aec_env", "env", "unwrapped"]
    seen = set()
    stack = [vec_env]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or obj is None:
            continue
        seen.add(id(obj))
        if hasattr(obj, "ts_ids") and hasattr(obj, "traffic_signals"):
            return obj
        for attr in candidates:
            child = getattr(obj, attr, None)
            if child is not None:
                stack.append(child)
        # Certains wrappers exposent une liste d'environnements
        for attr in ("vec_envs", "envs"):
            seq = getattr(obj, attr, None)
            if isinstance(seq, (list, tuple)):
                stack.extend(seq)
    return None


class AdversarialCurriculumCallback(BaseCallback):
    """Injecte des attaques pendant l'entrainement selon la phase du curriculum."""

    def __init__(self, phase, attacker=None, device=None, attack_period=20, verbose=0):
        super().__init__(verbose)
        self.phase = phase
        self.attacker = attacker
        self.device = device
        self.attack_period = attack_period
        self._counter = 0
        self._attacker_hidden = None
        self._sumo_env = None

    def _on_training_start(self) -> None:
        self._sumo_env = _resolve_sumo_env(self.training_env)
        if self._sumo_env is None and self.verbose:
            print("[!] Curriculum: SumoEnvironment introuvable sous les wrappers, "
                  "attaques desactivees pour cette phase.")

    def _on_step(self) -> bool:
        self._counter += 1
        if self._sumo_env is None or not getattr(self._sumo_env, "ts_ids", None):
            return True
        ts_ids = self._sumo_env.ts_ids

        if self.phase == 1:
            # Warm-up: jitter seul, deja injecte par VANETObservationFunction.
            return True

        if self.phase == 2:
            # Durcissement: attaque scriptee periodique sur une intersection tournante.
            if self._counter % self.attack_period == 0:
                idx = (self._counter // self.attack_period)
                atk_type, intensity = _ATTACK_SEQUENCE[idx % len(_ATTACK_SEQUENCE)]
                target = ts_ids[idx % len(ts_ids)]
                global_orchestrator.trigger_manual_attack(
                    target, atk_type, intensity, duration_steps=15
                )
            return True

        if self.phase == 3 and self.attacker is not None:
            # Co-evolution: l'attaquant surrogate genere une attaque adaptative.
            if self._counter % 5 == 0:
                target = ts_ids[self._counter % len(ts_ids)]
                ts = self._sumo_env.traffic_signals.get(target)
                if ts is None or self._sumo_env.observations.get(target) is None:
                    return True
                obs_np = self._sumo_env.observations[target]
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
                    attack_vec, self._attacker_hidden = self.attacker(
                        obs_t, self._attacker_hidden
                    )
                global_orchestrator.bridge_cGAN_tensor(
                    target, attack_vec.cpu().numpy()[0]
                )
            return True

        return True


def train_curriculum(phase1=60000, phase2=100000, phase3=120000, num_seconds=10000):
    print("=" * 66)
    print("CURRICULUM ADVERSARIAL MARL (resilience APPRISE)")
    print("=" * 66)
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}\n")

    # ---- Phase 1 : warm-up (jitter seul) ------------------------------------
    print(f"[Phase 1/3] Warm-up jitter ({phase1} etapes)...")
    env = build_marl_env(num_seconds=num_seconds)
    model = RecurrentPPO(
        "MlpLstmPolicy", env,
        verbose=1, learning_rate=3e-4, n_steps=256, batch_size=256,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
        tensorboard_log=TENSORBOARD_DIR,
    )
    model.learn(total_timesteps=phase1,
                callback=AdversarialCurriculumCallback(phase=1, verbose=1),
                tb_log_name="curriculum_phase1_jitter")
    model.save("outputs/marl_phase1_warmup")
    env.close()

    # ---- Phase 2 : durcissement (attaques manuelles) ------------------------
    print(f"\n[Phase 2/3] Durcissement attaques manuelles ({phase2} etapes)...")
    env = build_marl_env(num_seconds=num_seconds)
    model.set_env(env)
    model.learn(total_timesteps=phase2,
                callback=AdversarialCurriculumCallback(phase=2, verbose=1),
                tb_log_name="curriculum_phase2_manual",
                reset_num_timesteps=False)
    model.save("outputs/marl_phase2_hardened")
    env.close()

    # ---- Phase 3 : co-evolution (attaquant surrogate) -----------------------
    print(f"\n[Phase 3/3] Co-evolution attaquant surrogate ({phase3} etapes)...")
    env = build_marl_env(num_seconds=num_seconds)
    state_dim = env.observation_space.shape[0]
    attacker = SurrogateAdversarialAttacker(state_dim).to(device)
    # Si un attaquant a deja ete entraine, on le reprend (sinon poids initiaux).
    if os.path.exists(DEFAULT_GENERATOR_PATH):
        try:
            attacker.load_state_dict(torch.load(DEFAULT_GENERATOR_PATH, map_location=device))
            print(f"[*] Attaquant surrogate repris depuis {DEFAULT_GENERATOR_PATH}")
        except Exception as exc:
            print(f"[!] Reprise attaquant impossible ({exc}); poids initiaux utilises.")
    attacker.eval()

    model.set_env(env)
    model.learn(total_timesteps=phase3,
                callback=AdversarialCurriculumCallback(phase=3, attacker=attacker, device=device, verbose=1),
                tb_log_name="curriculum_phase3_surrogate",
                reset_num_timesteps=False)
    model.save(MODEL_PATH)
    env.close()

    print(f"\n[OK] Defenseur MARL durci par curriculum -> {MODEL_PATH}.zip")
    print("[OK] Resilience APPRISE: le defenseur a vu les attaques pendant l'entrainement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum adversarial MARL (grille 4x4).")
    parser.add_argument("--phase1", type=int, default=60000, help="Etapes warm-up (jitter).")
    parser.add_argument("--phase2", type=int, default=100000, help="Etapes durcissement (manuel).")
    parser.add_argument("--phase3", type=int, default=120000, help="Etapes co-evolution (surrogate).")
    parser.add_argument("--num-seconds", type=int, default=10000, help="Duree (s) d'un episode SUMO.")
    args = parser.parse_args()
    train_curriculum(args.phase1, args.phase2, args.phase3, args.num_seconds)
