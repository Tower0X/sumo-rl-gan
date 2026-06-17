import time
import torch
import numpy as np
import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import global_orchestrator, AttackType
from sumo_rl.environment.gan_attacker import load_generator_strict, GANLoadError
from sb3_contrib import RecurrentPPO
from shared_state import state

def run_simulation():
    state.reset()
    state.running = True
    state.add_log("🏁 Démarrage du bouclier urbain (Moteur SUMO)...", "info")
    
    # 1. Initialisation de l'environnement Grille 4x4
    try:
        env = sumo_rl.SumoEnvironment(
            net_file='sumo_rl/nets/4x4-Lucas/4x4.net.xml',
            route_file='sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
            use_gui=state.use_gui,
            num_seconds=20000, 
            delta_time=5,
            single_agent=True,
            observation_class=VANETObservationFunction,
            reward_fn='vanet',
            collision_action='warn'  # Collisions réellement mesurées
        )
    except Exception as e:
        state.add_log(f"❌ Erreur SUMO: {str(e)}", "error")
        state.running = False
        return

    ts_ids = env.ts_ids
    with state.lock:
        state.available_nodes = ts_ids
        if not state.target_node_id and ts_ids:
            state.target_node_id = ts_ids[0]

    # 2. Chargement du Défenseur RecurrentPPO
    state.add_log("[*] Chargement de la Matrice Neurale Recurrente (LSTM)...", "info")
    try:
        defender = RecurrentPPO.load("outputs/recurrent_urban_shield_4x4")
        # RecurrentPPO a besoin de son état caché initial
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
    except Exception as e:
        state.add_log(f"⚠️ Modèle récurrent non trouvé, basculement mode standard.", "warning")
        from stable_baselines3 import PPO
        try:
            defender = PPO.load("outputs/ppo_marl_4x4_model")
            lstm_states = None
        except:
            state.add_log("❌ Aucun modèle de défense trouvé.", "error")
            env.close()
            state.running = False
            return

    # 3. Chargement de l'Attaquant LSTM-GAN (STRICT: jamais de poids aléatoires)
    state.add_log("[*] Chargement du Neural Hacker (LSTM GAN)...", "info")
    state_dim = env.observation_space.shape[0]
    gan_loaded = False
    gan_hidden = None
    gan_hacker = None
    try:
        gan_hacker = load_generator_strict(state_dim)
        gan_loaded = True
    except GANLoadError as exc:
        state.add_log(f"⚠️ GAN non chargé ({exc}). Mode adversarial désactivé.", "warning")

    obs, info = env.reset()
    done = False
    
    state.add_log("🚀 Supervision Urbaine (16 agents) active.", "info")

    while not done and not state.should_stop:
        with state.lock:
            current_mode = state.mode
            manual_atk = state.manual_attack_type
            manual_virulence = state.manual_virulence
            target_ts = state.target_node_id

        # --- GESTION DES ATTAQUES ---
        active_name = "Aucune"
        active_intensity = 0.0

        if current_mode == "defense_only":
            global_orchestrator.active_attacks.clear()
            for tid in ts_ids: env.traffic_signals[tid].comm_ok = True
            
        elif current_mode == "manual_attack":
            if manual_atk != AttackType.NONE and target_ts:
                global_orchestrator.trigger_manual_attack(target_ts, manual_atk, manual_virulence, duration_steps=1)
                active_name = f"{manual_atk.name} sur {target_ts}"
                active_intensity = manual_virulence
                
        elif current_mode == "adversarial_gan":
            if not gan_loaded:
                state.add_log("⛔ Mode GAN demandé mais aucun GAN entraîné chargé. Attaque ignorée.", "error")
            if gan_loaded:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
                    attack_vector_tensor, gan_hidden = gan_hacker(state_tensor, gan_hidden)
                    attack_vector = attack_vector_tensor.numpy()[0]
                
                # Le GAN choisit une cible ou on garde la cible dashboard
                global_orchestrator.bridge_cGAN_tensor(target_ts, attack_vector)
                if target_ts in global_orchestrator.active_attacks:
                    atk = global_orchestrator.active_attacks[target_ts]
                    active_name = f"GAN: {atk['type'].name}"
                    active_intensity = atk["intensity"]

        # --- DÉCISION PPO (AVEC MÉMOIRE) ---
        if lstm_states is not None:
            action, lstm_states = defender.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = defender.predict(obs, deterministic=True)

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # --- MISE À JOUR ÉTAT ---
        with state.lock:
            state.step += 1
            state.active_attack_name = active_name
            state.active_attack_intensity = active_intensity
            state.current_reward = reward
            state.reward_history.append(reward)
            state.waiting_time_history.append(step_info.get('system_total_waiting_time', 0))
            latency = float(np.abs(next_obs[-2]) * 10)
            state.latency_history.append(latency)
            mid = len(next_obs) // 2
            total_queues = int(np.sum(next_obs[mid:-2]) * 10)
            state.queue_history.append(total_queues)

        obs = next_obs
        done = terminated or truncated
        time.sleep(0.05)

    env.close()
    with state.lock: state.running = False
    state.add_log("🏁 Simulation arrêtée.", "info")
