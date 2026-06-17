import csv
import time
import os
import numpy as np
from enum import Enum

class AttackType(Enum):
    NONE = 0
    TEMPORAL_DOS = 1        # Latence de base
    DATA_POISONING = 2      # Modification des capteurs
    GHOST_VEHICLES = 3      # Sybil classique (Densité)
    JAMMER = 4              # Coupure totale (Brouilleur)
    FLOODING_DDOS = 5       # Saturation de requêtes (Jitter intense)
    SLOWLORIS_DDOS = 6      # Connexions maintenues (Lag persistant)
    POSITION_JITTER = 7     # Bruit sur les positions (Sybil cinématique)

class TargetType(Enum):
    INTERSECTION = "Intersection"
    VEHICLE = "Vehicle"
    RADIUS = "Radius"


def compute_obs_layout(ts):
    """Compute the EXACT feature boundaries of a VANET observation vector.

    The VANET observation produced by ``VANETObservationFunction`` is structured as::

        [ phase_one_hot (num_green_phases) | min_green (1) |
          density (n_lanes) | queue (n_lanes) | latency (1) | comm_flag (1) ]

    Relying on ``len(obs)//2`` (as the previous implementation did) does NOT land on
    these boundaries because the one-hot phase block and the ``min_green`` flag shift
    every index. We therefore derive the offsets directly from the traffic signal so
    that each attack poisons exactly the features it claims to target.

    Returns:
        dict with keys ``offset_density``, ``offset_queue``, ``n_lanes``,
        ``latency_idx`` and ``comm_idx``.
    """
    num_green_phases = int(getattr(ts, "num_green_phases", 0))
    n_lanes = len(getattr(ts, "lanes", []))
    offset_density = num_green_phases + 1  # +1 for the min_green flag
    offset_queue = offset_density + n_lanes
    return {
        "offset_density": offset_density,
        "offset_queue": offset_queue,
        "n_lanes": n_lanes,
        "latency_idx": -2,
        "comm_idx": -1,
    }


class CyberPhysicalAttackOrchestrator:
    """
    Le Cœur de l'Arsenal. Ce contrôleur intercepte les flux de données entre
    SUMO (la physique) et l'Agent PPO (l'IA), permettant d'injecter des
    failles de sécurité sophistiquées. Pilotable à la main ou par un GAN.
    """
    def __init__(self, log_path="outputs/attack_log.csv"):
        self.active_attacks = {}  # Stocke les attaques en cours {target_id: attack_config}
        self.log_path = log_path
        self._init_logger()

    def _init_logger(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Target_ID", "Target_Type", "Attack_Type", "Intensity", "PPO_Reward_Impact"])

    def log_attack_impact(self, target_id, target_type, attack_type, intensity, reward_impact):
        """ Enregistre la Télémétrie pour entraîner / analyser l'attaquant """
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            tt = target_type.value if isinstance(target_type, TargetType) else target_type
            at = attack_type.name if isinstance(attack_type, AttackType) else attack_type
            writer.writerow([time.time(), target_id, tt, at, intensity, reward_impact])

    # ==========================================
    # MOTEUR DE GÉNÉRATION : MODE MANUAL OVERRIDE
    # ==========================================
    def trigger_manual_attack(self, target_id, attack_type: AttackType, intensity: float, duration_steps: int = 10, target_type=TargetType.INTERSECTION):
        """
        Déclenche une attaque ciblée.
        intensity: Float entre 0.0 (Faible) et 1.0 (Destruction Totale)
        """
        self.active_attacks[target_id] = {
            "type": attack_type,
            "intensity": float(np.clip(intensity, 0.0, 1.0)),
            "remaining_steps": duration_steps,
            "target_type": target_type
        }
        print(f"\n[⚠️ ALERTE CYBER] Attaque {attack_type.name} (Force: {intensity*100:.0f}%) lancée sur {target_type.value} {target_id} !")

    # ==========================================
    # MOTEUR DE GÉNÉRATION : cGAN INPUT BRIDGE
    # ==========================================
    def bridge_cGAN_tensor(self, target_id, gan_output_tensor):
        """
        Traduit la sortie brute du réseau PyTorch (GAN LSTM) en commandes physiques.
        gan_output_tensor est composé de [P_None, ..., P_Attack7, Intensity] (9 éléments)
        """
        # On sépare les probabilités du type d'attaque (8 premiers éléments)
        attack_probs = gan_output_tensor[:8]
        intensity = float(gan_output_tensor[8])  # Le dernier élément est l'intensité

        # On prend l'index de l'attaque la plus probable
        attack_idx = int(np.argmax(attack_probs))
        attack_type = AttackType(attack_idx)

        # Le GAN décide d'attaquer si la confiance est suffisante
        if attack_type != AttackType.NONE and intensity > 0.05:
            self.trigger_manual_attack(target_id, attack_type, intensity, duration_steps=3)

    # ==========================================
    # L'ARSENAL : INJECTION DANS L'OBSERVATION
    # ==========================================
    def corrupt_observation(self, ts, raw_obs):
        """
        Fonction maîtresse. Intercepte le vecteur d'observation parfait et l'empoisonne
        selon les attaques en cours avant de le donner au PPO. Modifie également le statut
        ``comm_ok`` du TrafficSignal pour impacter la récompense (lien Fail-Safe).

        Garanties:
        - Chaque attaque ne perturbe QUE les features qu'elle prétend cibler
          (indexation exacte via ``compute_obs_layout``).
        - Le vecteur retourné est TOUJOURS clippé dans [0, 1]: il ne peut jamais
          violer l'espace d'observation Box[0, 1].
        """
        ts_id = ts.id
        if ts_id not in self.active_attacks:
            ts.comm_ok = True  # Restaure la santé si pas d'attaque
            return raw_obs  # Réseau sain, aucune modification

        attack = self.active_attacks[ts_id]
        if attack["remaining_steps"] <= 0:
            del self.active_attacks[ts_id]
            ts.comm_ok = True
            print(f"[✅ RECOVERY] Fin de l'attaque sur {ts_id}. Réseau restauré.")
            return raw_obs

        # Décrémentation du timer
        attack["remaining_steps"] -= 1

        corrupted_obs = np.copy(raw_obs).astype(np.float32)
        intensity = attack["intensity"]
        attack_type = attack["type"]

        # --- Frontières EXACTES des features (fini le len//2 approximatif) ---
        layout = compute_obs_layout(ts)
        od = layout["offset_density"]
        oq = layout["offset_queue"]
        n_lanes = layout["n_lanes"]
        lat = layout["latency_idx"]
        comm = layout["comm_idx"]
        density_slice = slice(od, od + n_lanes)
        queue_slice = slice(oq, oq + n_lanes)

        # -----------------------------------------------------
        # 1. TEMPORAL DoS / FLOODING / SLOWLORIS -> perturbent la LATENCE uniquement
        # -----------------------------------------------------
        if attack_type == AttackType.TEMPORAL_DOS:
            corrupted_obs[lat] += np.random.normal(0, intensity * 0.5)
            if intensity > 0.9:
                ts.comm_ok = False

        elif attack_type == AttackType.FLOODING_DDOS:
            corrupted_obs[lat] += np.random.uniform(0, intensity)
            if np.random.random() < intensity:
                ts.comm_ok = False

        elif attack_type == AttackType.SLOWLORIS_DDOS:
            corrupted_obs[lat] += intensity

        # -----------------------------------------------------
        # 2. DATA POISONING -> cache les embouteillages: bloc QUEUE uniquement
        # -----------------------------------------------------
        elif attack_type == AttackType.DATA_POISONING:
            corrupted_obs[queue_slice] = corrupted_obs[queue_slice] * (1.0 - intensity)

        # -----------------------------------------------------
        # 3. GHOST VEHICLES (Sybil densité) -> bloc DENSITY uniquement
        # -----------------------------------------------------
        elif attack_type == AttackType.GHOST_VEHICLES:
            corrupted_obs[density_slice] = corrupted_obs[density_slice] + intensity

        # POSITION JITTER -> bruit gaussien sur la DENSITY uniquement
        elif attack_type == AttackType.POSITION_JITTER:
            jitter = np.random.normal(0, intensity, n_lanes)
            corrupted_obs[density_slice] = corrupted_obs[density_slice] + jitter

        # -----------------------------------------------------
        # 4. JAMMER (Brouilleur total) -> latence max, comm coupée (jamais 99.0)
        # -----------------------------------------------------
        elif attack_type == AttackType.JAMMER:
            ts.comm_ok = False
            corrupted_obs[lat] = 1.0   # latence maximale normalisée
            corrupted_obs[comm] = 0.0  # communication coupée

        # --- GARANTIE D'INTÉGRITÉ: ne JAMAIS violer Box[0, 1] ---
        corrupted_obs = np.clip(corrupted_obs, 0.0, 1.0).astype(np.float32)

        # --- TÉLÉMÉTRIE (auparavant morte) ---
        # On journalise l'impact mesurable de l'attaque sur l'observation.
        reward_impact = float(np.linalg.norm(corrupted_obs - raw_obs))
        self.log_attack_impact(
            ts_id, attack.get("target_type", TargetType.INTERSECTION),
            attack_type, intensity, reward_impact,
        )

        return corrupted_obs

# Instance globale pour un accès partagé entre toutes les intersections
global_orchestrator = CyberPhysicalAttackOrchestrator()
