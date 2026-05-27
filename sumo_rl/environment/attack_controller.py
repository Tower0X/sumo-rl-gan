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

class CyberPhysicalAttackOrchestrator:
    """
    Le Cœur de l'Arsenal. Ce contrôleur intercepte les flux de données entre
    SUMO (la physique) et l'Agent PPO (l'IA), permettant d'injecter des
    failles de sécurité sophistiquées. Pilatable à la main ou par un GAN.
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
        """ Enregistre la Télémétrie pour entraîner le futur cGAN """
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), target_id, target_type, attack_type.name, intensity, reward_impact])

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
            "intensity": np.clip(intensity, 0.0, 1.0),
            "remaining_steps": duration_steps,
            "target_type": target_type
        }
        print(f"\n[⚠️ ALERTE CYBER] Attaque {attack_type.name} (Force: {intensity*100}%) lancée sur {target_type.value} {target_id} !")

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
        intensity = gan_output_tensor[8] # Le dernier élément est l'intensité
        
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
        Fonction maîtresse. Intercepte le vecteur d'observation parfait 
        et l'empoisonne selon les attaques en cours avant de le donner au PPO.
        Modifie également le statut du TrafficSignal pour impacter la récompense.
        """
        ts_id = ts.id
        if ts_id not in self.active_attacks:
            ts.comm_ok = True  # Restaure la santé si pas d'attaque
            return raw_obs # Réseau sain, aucune modification
            
        attack = self.active_attacks[ts_id]
        if attack["remaining_steps"] <= 0:
            del self.active_attacks[ts_id]
            ts.comm_ok = True
            print(f"[✅ RECOVERY] Fin de l'attaque sur {ts_id}. Réseau restauré.")
            return raw_obs
            
        # Décrémentation du timer
        attack["remaining_steps"] -= 1
        
        corrupted_obs = np.copy(raw_obs)
        intensity = attack["intensity"]
        
        # -----------------------------------------------------
        # 1. TEMPORAL DoS (Latence & Brouillage)
        # -----------------------------------------------------
        # 1. TEMPORAL DoS (Latence de base) / FLOODING / SLOWLORIS
        if attack["type"] == AttackType.TEMPORAL_DOS:
            noise = np.random.normal(0, intensity * 2.0) 
            corrupted_obs[-2] += noise
            if intensity > 0.9: ts.comm_ok = False

        elif attack["type"] == AttackType.FLOODING_DDOS:
            noise = np.random.uniform(0, intensity * 10.0)
            corrupted_obs[-2] += noise
            if np.random.random() < intensity: ts.comm_ok = False

        elif attack["type"] == AttackType.SLOWLORIS_DDOS:
            corrupted_obs[-2] += intensity * 8.0

        # -----------------------------------------------------
        # 2. DATA POISONING (Déni de réalité - Fausse fluidité)
        # -----------------------------------------------------
        elif attack["type"] == AttackType.DATA_POISONING:
            # Le but est de cacher les embouteillages à l'IA.
            # On divise les valeurs de 'queue' par l'intensité de l'attaque.
            # (Hypothèse: les indices de queue sont dans la 2ème moitié du vecteur)
            mid = len(corrupted_obs) // 2
            corrupted_obs[mid:-2] = corrupted_obs[mid:-2] * (1.0 - intensity)

        # -----------------------------------------------------
        # 3. GHOST VEHICLES (Attaque Sybil)
        # -----------------------------------------------------
        # 3. GHOST VEHICLES (Sybil de densité) / POSITION JITTER
        elif attack["type"] == AttackType.GHOST_VEHICLES:
            mid = len(corrupted_obs) // 2
            corrupted_obs[1:mid] = np.clip(corrupted_obs[1:mid] + intensity, 0.0, 1.0)

        elif attack["type"] == AttackType.POSITION_JITTER:
            mid = len(corrupted_obs) // 2
            jitter = np.random.normal(0, intensity, mid-1)
            corrupted_obs[1:mid] = np.clip(corrupted_obs[1:mid] + jitter, 0.0, 1.0)

        # 4. JAMMER (Brouilleur total)
        elif attack["type"] == AttackType.JAMMER:
            ts.comm_ok = False
            corrupted_obs[-1] = 0.0
            corrupted_obs[-2] = 99.0

        return corrupted_obs

# Instance globale pour un accès partagé entre toutes les intersections
global_orchestrator = CyberPhysicalAttackOrchestrator()

