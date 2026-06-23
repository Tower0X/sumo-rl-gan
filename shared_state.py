import threading
from sumo_rl.environment.attack_controller import AttackType

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.should_stop = False
        
        # Paramètres de simulation configurables
        self.mode = "defense_only"  # "defense_only", "manual_attack", "adversarial_gan"
        self.city_map = "Ville par défaut (4x4 Lucas)"
        self.manual_attack_type = AttackType.NONE
        self.manual_virulence = 0.5
        self.target_node_id = ""
        self.available_nodes = []
        self.use_gui = False
        
        # Métriques et Historiques pour le Dashboard
        self.step = 0
        self.active_attack_name = "Aucune"
        self.active_attack_intensity = 0.0
        self.current_reward = 0.0

        # Telemetrie du canal physique (Lot E)
        self.total_ghosts_spawned = 0
        self.phase_frozen = False
        self.target_comm_ok = True       # V2X de l'intersection ciblée
        self.n_intersections_attacked = 0 # Compteur d'intersections attaquées
        self.target_reward = 0.0          # Reward de la cible (pas la moyenne)

        self.reward_history = []
        self.target_reward_history = []   # Historique reward cible seule
        self.waiting_time_history = []
        self.latency_history = []
        self.queue_history = []
        self.logs = []

    def reset(self):
        with self.lock:
            self.running = False
            self.should_stop = False
            self.step = 0
            self.active_attack_name = "Aucune"
            self.active_attack_intensity = 0.0
            self.current_reward = 0.0
            self.total_ghosts_spawned = 0
            self.phase_frozen = False
            self.target_comm_ok = True
            self.n_intersections_attacked = 0
            self.target_reward = 0.0
            self.reward_history = []
            self.target_reward_history = []
            self.waiting_time_history = []
            self.latency_history = []
            self.queue_history = []
            self.logs = []

    def add_log(self, text, type="info"):
        with self.lock:
            self.logs.append({"step": self.step, "text": text, "type": type})
            if len(self.logs) > 50:
                self.logs.pop(0)

# Instance unique partagée
state = SharedState()
