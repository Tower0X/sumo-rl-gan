import time
import os
import traceback
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import global_orchestrator
from sumo_rl.environment.gan_attacker import init_gan_components

def train_gan_recurrent():
    print("======================================================")
    print("⚔️ ARÈNE ADVERSARIALE 2.1 : ENTRAÎNEMENT DU LSTM-GAN")
    print("======================================================\n")

    env = None
    try:
        # 1. Init Environnement 4x4
        print("[*] Démarrage du moteur urbain SUMO...")
        env = sumo_rl.SumoEnvironment(
            net_file='sumo_rl/nets/4x4-Lucas/4x4.net.xml',
            route_file='sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
            use_gui=False, 
            num_seconds=10000, 
            delta_time=5,
            single_agent=True,
            observation_class=VANETObservationFunction,
            reward_fn='vanet'
        )

        # 2. Chargement du Défenseur
        print("[*] Déploiement de l'Agent de Défense (RecurrentPPO)...")
        model_path = os.path.join("outputs", "recurrent_urban_shield_4x4")
        try:
            defender_model = RecurrentPPO.load(model_path)
        except:
            defender_model = RecurrentPPO.load(model_path + ".zip")

        # 3. Initialisation du GAN
        state_dim = env.observation_space.shape[0]
        generator, discriminator, opt_G, opt_D, device = init_gan_components(state_dim)
        mse_loss = nn.MSELoss()
        
        num_episodes = 25
        print(f"\n[*] 🚀 DÉBUT DU DUEL ({num_episodes} Épisodes)...")
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            step = 0
            
            gan_gen_hidden = None
            gan_disc_hidden = None
            ppo_lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            ep_gan_reward = 0
            
            while not done:
                # --- PHASE 1 : ATTAQUE ---
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
                
                # IMPORTANT: detacher les états cachés pour éviter l'accumulation infinie de gradient
                if gan_gen_hidden is not None:
                    gan_gen_hidden = (gan_gen_hidden[0].detach(), gan_gen_hidden[1].detach())
                
                attack_vector_tensor, gan_gen_hidden = generator(state_tensor, gan_gen_hidden)
                attack_array = attack_vector_tensor.cpu().detach().numpy()[0]
                
                ts_id = env.ts_ids[step % len(env.ts_ids)]
                global_orchestrator.bridge_cGAN_tensor(ts_id, attack_array)
                
                # --- PHASE 2 : RÉPONSE ---
                action, ppo_lstm_states = defender_model.predict(
                    obs, state=ppo_lstm_states, episode_start=episode_starts, deterministic=True
                )
                episode_starts = np.zeros((1,), dtype=bool)
                
                next_obs, ppo_reward, terminated, truncated, _ = env.step(action)
                real_gan_reward = -ppo_reward
                ep_gan_reward += real_gan_reward
                
                # --- PHASE 3 : DISCRIMINATEUR (JUMEAU) ---
                if gan_disc_hidden is not None:
                    gan_disc_hidden = (gan_disc_hidden[0].detach(), gan_disc_hidden[1].detach())
                
                opt_D.zero_grad()
                pred_reward, gan_disc_hidden = discriminator(state_tensor, attack_vector_tensor.detach(), gan_disc_hidden)
                target_reward = torch.FloatTensor([[real_gan_reward]]).to(device)
                
                loss_D = mse_loss(pred_reward, target_reward)
                loss_D.backward(retain_graph=True)
                opt_D.step()
                
                # --- PHASE 4 : GÉNÉRATEUR (ATTAQUANT) ---
                opt_G.zero_grad()
                # On réutilise le hidden disc mis à jour pour évaluer G
                pred_reward_for_G, _ = discriminator(state_tensor, attack_vector_tensor, (gan_disc_hidden[0].detach(), gan_disc_hidden[1].detach()))
                loss_G = -pred_reward_for_G.mean()
                loss_G.backward()
                opt_G.step()
                
                obs = next_obs
                done = terminated or truncated
                step += 1
                
            print(f"Épisode {episode+1}/{num_episodes} | Dommage Urbain: {ep_gan_reward:.2f}")

        # Sauvegarde
        os.makedirs("outputs/gan", exist_ok=True)
        torch.save(generator.state_dict(), "outputs/gan/generator_model_lstm.pth")
        print("\n[✅] ENTRAÎNEMENT TERMINÉ. Le Hacker est prêt !")
        
    except Exception as e:
        print(f"\n[❌] ERREUR CRITIQUE : {e}")
        traceback.print_exc()
    finally:
        if env:
            env.close()

if __name__ == "__main__":
    train_gan_recurrent()
