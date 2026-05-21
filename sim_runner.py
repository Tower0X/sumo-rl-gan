import time
import torch
import numpy as np
import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import global_orchestrator, AttackType
from sumo_rl.environment.gan_attacker import Generator
from stable_baselines3 import PPO
from shared_state import state

def run_simulation():
    state.reset()
    state.running = True
    state.add_log("🏁 Démarrage du moteur physique SUMO...", "info")
    
    # 1. Initialisation de l'environnement
    try:
        env = sumo_rl.SumoEnvironment(
            net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
            route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
            use_gui=state.use_gui,
            num_seconds=10000, 
            delta_time=5,
            single_agent=True,
            observation_class=VANETObservationFunction,
            reward_fn='vanet'
        )
    except Exception as e:
        state.add_log(f"❌ Erreur SUMO: {str(e)}", "error")
        state.running = False
        return

    ts_id = env.ts_ids[0]
    
    # 2. Chargement du Défenseur PPO
    state.add_log("[*] Chargement de l'Agent de Défense (PPO)...", "info")
    try:
        defender = PPO.load("outputs/ppo_vanet_model")
    except Exception as e:
        state.add_log(f"❌ Impossible de charger PPO: {str(e)}", "error")
        env.close()
        state.running = False
        return

    # 3. Chargement de l'Attaquant cGAN
    state.add_log("[*] Chargement du Neural Hacker (cGAN)...", "info")
    state_dim = env.observation_space.shape[0]
    gan_hacker = Generator(state_dim)
    gan_loaded = False
    try:
        gan_hacker.load_state_dict(torch.load("outputs/gan/generator_model.pth"))
        gan_hacker.eval()
        gan_loaded = True
    except Exception as e:
        state.add_log(f"⚠️ Impossible de charger GAN (mode Duel désactivé): {str(e)}", "warning")

    obs, info = env.reset()
    done = False
    
    state.add_log("🚀 Co-Simulation activée en arrière-plan.", "info")

    while not done and not state.should_stop:
        # Lire la configuration partagée (avec lock)
        with state.lock:
            current_mode = state.mode
            manual_atk = state.manual_attack_type
            manual_virulence = state.manual_virulence

        # Appliquer les attaques selon le mode
        if current_mode == "defense_only":
            global_orchestrator.active_attacks.clear()
            env.traffic_signals[ts_id].comm_ok = True
            active_name = "Aucune"
            active_intensity = 0.0
            
        elif current_mode == "manual_attack":
            if manual_atk != AttackType.NONE:
                global_orchestrator.trigger_manual_attack(
                    ts_id, 
                    manual_atk, 
                    manual_virulence, 
                    duration_steps=1
                )
                active_name = manual_atk.name
                active_intensity = manual_virulence
            else:
                global_orchestrator.active_attacks.clear()
                env.traffic_signals[ts_id].comm_ok = True
                active_name = "Aucune"
                active_intensity = 0.0
                
        elif current_mode == "adversarial_gan":
            if gan_loaded:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    attack_vector = gan_hacker(state_tensor).numpy()[0]
                
                global_orchestrator.bridge_cGAN_tensor(ts_id, attack_vector)
                
                # Récupérer l'attaque injectée
                if ts_id in global_orchestrator.active_attacks:
                    atk = global_orchestrator.active_attacks[ts_id]
                    active_name = atk["type"].name
                    active_intensity = atk["intensity"]
                else:
                    active_name = "Aucune"
                    active_intensity = 0.0
            else:
                active_name = "Indisponible (Erreur GAN)"
                active_intensity = 0.0

        # Décision de défense (PPO) sur l'observation potentiellement corrompue
        action, _ = defender.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Enregistrer les métriques (avec lock)
        with state.lock:
            state.step += 1
            state.active_attack_name = active_name
            state.active_attack_intensity = active_intensity
            state.current_reward = reward
            
            state.reward_history.append(reward)
            state.waiting_time_history.append(step_info.get('system_total_waiting_time', 0))
            
            # Latence standard + gigue (normalement l'avant-dernier élément du vecteur d'obs)
            latency = float(np.abs(next_obs[-2]) * 10)
            state.latency_history.append(latency)
            
            # Queues accumulées sur toutes les voies (2ème moitié du vecteur hors latence/comm)
            mid = len(next_obs) // 2
            total_queues = int(np.sum(next_obs[mid:-2]) * 10)
            state.queue_history.append(total_queues)

        # Journalisation des événements notables
        if state.step % 10 == 0:
            if active_name != "Aucune":
                state.add_log(f"⚠️ Attaque active: {active_name} | Virulence: {active_intensity*100:.1f}% | Récompense: {reward:.2f}", "warning")
            else:
                state.add_log(f"🟢 Trafic sain | Temps d'attente: {step_info.get('system_total_waiting_time', 0):.1f}s | Récompense: {reward:.2f}", "info")

        obs = next_obs
        done = terminated or truncated
        
        # Ralentir la simulation pour permettre le rendu fluide du dashboard
        time.sleep(0.15 if state.use_gui else 0.08)

    env.close()
    with state.lock:
        state.running = False
    state.add_log("🏁 Simulation arrêtée. Connexion SUMO fermée.", "info")
