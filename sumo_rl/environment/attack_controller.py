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


# Familles d'attaques exposées à l'UI (segmentation Lot E). Source de vérité
# unique partagée par le dashboard et l'orchestrateur pour éviter toute
# divergence label UI <-> Enum.
ATTACK_FAMILY_LABELS = {
    AttackType.NONE: "Aucune",
    AttackType.JAMMER: "Brouilleur (Coupure Totale)",
    AttackType.FLOODING_DDOS: "DoS - Flooding (Saturation)",
    AttackType.SLOWLORIS_DDOS: "DoS - Slowloris (Lag Persistant)",
    AttackType.GHOST_VEHICLES: "Sybil - Ghost Vehicles (Densité)",
    AttackType.POSITION_JITTER: "Sybil - Position Jitter (Cinématique)",
    AttackType.TEMPORAL_DOS: "Temporal DoS (Latence Variable)",
    AttackType.DATA_POISONING: "Data Poisoning (Capteurs)",
}

# Couleur RGBA des véhicules fantômes injectés (ROUGE vif, visibles en GUI).
GHOST_VEHICLE_COLOR = (255, 0, 0, 255)
GHOST_VEHICLE_TYPE_ID = "vanet_ghost_attacker"


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

    Deux canaux d'attaque COMPLÉMENTAIRES (Lot E):
      1. PERCEPTUEL (``corrupt_observation``) : empoisonne le vecteur
         d'observation lu par le PPO (densité, files, latence, comm).
      2. PHYSIQUE (``apply_physical_attack``) : agit directement sur SUMO via
         TraCI (véhicules fantômes ROUGES, gel/forçage de phase), garantissant
         un effet VISIBLE et MESURABLE même face à un PPO durci par curriculum.
    """
    def __init__(self, log_path="outputs/attack_log.csv"):
        self.active_attacks = {}  # {target_id: attack_config}
        self.log_path = log_path
        # Compteur global pour nommer de manière unique les véhicules fantômes.
        self._ghost_counter = 0
        # Mémorise les types de véhicule déjà créés par connexion SUMO.
        self._ghost_type_ready = set()
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
        duration_steps: durée de vie de l'attaque en LECTURES d'observation
            (canal perceptuel). Le canal physique vit tant que l'attaque est
            ré-armée par le runner.
        """
        existing = self.active_attacks.get(target_id)
        # Re-arming d'une attaque identique déjà active: on prolonge sans spammer le log.
        if existing and existing["type"] == attack_type:
            existing["intensity"] = float(np.clip(intensity, 0.0, 1.0))
            existing["remaining_steps"] = max(existing["remaining_steps"], duration_steps)
            existing["target_type"] = target_type
            return
        self.active_attacks[target_id] = {
            "type": attack_type,
            "intensity": float(np.clip(intensity, 0.0, 1.0)),
            "remaining_steps": duration_steps,
            "target_type": target_type,
        }
        print(f"\n[!] ALERTE CYBER: Attaque {attack_type.name} (Force: {intensity*100:.0f}%) lancee sur {target_type.value} {target_id} !")

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
            print(f"[OK] RECOVERY: Fin de l'attaque sur {ts_id}. Reseau restaure.")
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
        # 1. TEMPORAL DoS / FLOODING / SLOWLORIS -> perturbent la LATENCE ET comm_ok
        # -----------------------------------------------------
        if attack_type == AttackType.TEMPORAL_DOS:
            corrupted_obs[lat] += np.random.normal(0.3, intensity * 0.5)
            # Coupe la communication proportionnellement à l'intensité
            if intensity > 0.5 or np.random.random() < intensity * 0.7:
                ts.comm_ok = False

        elif attack_type == AttackType.FLOODING_DDOS:
            # Saturation massive de la latence
            corrupted_obs[lat] = min(1.0, corrupted_obs[lat] + intensity * 0.8)
            # Forte probabilité de couper la communication
            if np.random.random() < intensity * 0.9:
                ts.comm_ok = False

        elif attack_type == AttackType.SLOWLORIS_DDOS:
            # Lag persistant et croissant
            corrupted_obs[lat] = min(1.0, corrupted_obs[lat] + intensity * 0.6)
            # Communication dégradée sous forte intensité
            if intensity > 0.6:
                ts.comm_ok = False

        # -----------------------------------------------------
        # 2. DATA POISONING -> cache les embouteillages: bloc QUEUE uniquement
        # -----------------------------------------------------
        elif attack_type == AttackType.DATA_POISONING:
            corrupted_obs[queue_slice] = corrupted_obs[queue_slice] * (1.0 - intensity)

        # -----------------------------------------------------
        # 3. GHOST VEHICLES (Sybil densité) -> bloc DENSITY uniquement
        # -----------------------------------------------------
        elif attack_type == AttackType.GHOST_VEHICLES:
            corrupted_obs[density_slice] = np.clip(
                corrupted_obs[density_slice] + intensity * 0.8, 0.0, 1.0
            )

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

    # ==========================================
    # L'ARSENAL : CANAL PHYSIQUE (TraCI) -- Lot E
    # ==========================================
    def _ensure_ghost_vtype(self, sumo, conn_label):
        """Crée (une seule fois par connexion) un type de véhicule ROUGE.

        Le type fantôme copie le comportement d'une voiture standard mais est
        peint en rouge vif pour être immédiatement identifiable dans SUMO-GUI.
        """
        if conn_label in self._ghost_type_ready:
            return
        try:
            existing = set(sumo.vehicletype.getIDList())
            if GHOST_VEHICLE_TYPE_ID not in existing:
                # Copie d'un type existant pour hériter de paramètres valides.
                base_type = "DEFAULT_VEHTYPE" if "DEFAULT_VEHTYPE" in existing else (
                    next(iter(existing)) if existing else None
                )
                if base_type is not None:
                    sumo.vehicletype.copy(base_type, GHOST_VEHICLE_TYPE_ID)
                else:
                    sumo.vehicletype.add(GHOST_VEHICLE_TYPE_ID)
                sumo.vehicletype.setColor(GHOST_VEHICLE_TYPE_ID, GHOST_VEHICLE_COLOR)
            self._ghost_type_ready.add(conn_label)
        except Exception as exc:
            print(f"[WARN] Impossible de créer le type fantôme: {exc}")

    def _spawn_ghost_vehicles(self, ts, sumo, n_ghosts):
        """Injecte ``n_ghosts`` véhicules ROUGES sur les voies entrantes du feu.

        Ce sont de VRAIS véhicules SUMO: ils occupent l'espace, sont vus par les
        capteurs (densité/queue réelles), et provoquent de vrais bouchons -> l'effet
        Sybil devient physiquement observable, pas seulement perceptuel.
        """
        conn_label = getattr(sumo, "label", "default")
        self._ensure_ghost_vtype(sumo, conn_label)
        lanes = list(getattr(ts, "lanes", []))
        if not lanes:
            return 0
        spawned = 0
        for i in range(n_ghosts):
            lane = lanes[i % len(lanes)]
            try:
                edge = lane.rsplit("_", 1)[0]
                route_id = f"ghost_route_{edge}"
                # Crée une route minimale (1 edge) si absente.
                try:
                    if route_id not in sumo.route.getIDList():
                        sumo.route.add(route_id, [edge])
                except Exception:
                    sumo.route.add(route_id, [edge])
                veh_id = f"GHOST_{ts.id}_{self._ghost_counter}"
                self._ghost_counter += 1
                sumo.vehicle.add(
                    veh_id,
                    route_id,
                    typeID=GHOST_VEHICLE_TYPE_ID,
                    departLane="free",
                    departSpeed="0",
                )
                sumo.vehicle.setColor(veh_id, GHOST_VEHICLE_COLOR)
                spawned += 1
            except Exception:
                # Voie pleine / insertion refusée: on ignore ce fantôme.
                continue
        return spawned

    def apply_physical_attack(self, ts, sumo):
        """Applique l'effet PHYSIQUE d'une attaque active via TraCI.

        À appeler par le runner APRÈS le step (quand SUMO est avancé) et AVANT la
        prochaine décision. Complète le canal perceptuel pour garantir un effet
        visible/mesurable face à un PPO durci.

        - GHOST_VEHICLES / POSITION_JITTER -> injecte des véhicules ROUGES réels.
        - JAMMER / FLOODING_DDOS / SLOWLORIS_DDOS / TEMPORAL_DOS -> gèle la phase
          courante (l'agent perd le contrôle physique du feu pendant l'attaque).
        - DATA_POISONING -> purement perceptuel (aucune action physique).

        Retourne un dict d'effets appliqués (pour la télémétrie dashboard).
        """
        effects = {"ghosts_spawned": 0, "phase_frozen": False}
        if sumo is None or ts.id not in self.active_attacks:
            return effects
        attack = self.active_attacks[ts.id]
        attack_type = attack["type"]
        intensity = attack["intensity"]

        if attack_type in (AttackType.GHOST_VEHICLES, AttackType.POSITION_JITTER):
            # Nombre de fantômes proportionnel à l'intensité (1 à 6 par step).
            n_ghosts = max(1, int(round(intensity * 6)))
            effects["ghosts_spawned"] = self._spawn_ghost_vehicles(ts, sumo, n_ghosts)

        elif attack_type in (
            AttackType.JAMMER,
            AttackType.FLOODING_DDOS,
            AttackType.SLOWLORIS_DDOS,
            AttackType.TEMPORAL_DOS,
        ):
            # Gel de la phase: on ré-impose l'état courant, neutralisant l'action
            # de l'agent. Probabilité de gel proportionnelle à l'intensité pour
            # les DoS partiels; gel certain pour le Jammer.
            freeze = attack_type == AttackType.JAMMER or np.random.random() < intensity
            if freeze:
                try:
                    current_state = sumo.trafficlight.getRedYellowGreenState(ts.id)
                    sumo.trafficlight.setRedYellowGreenState(ts.id, current_state)
                    effects["phase_frozen"] = True
                except Exception:
                    pass

        return effects

# Instance globale pour un accès partagé entre toutes les intersections
global_orchestrator = CyberPhysicalAttackOrchestrator()
