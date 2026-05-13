import time
import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

import sumo_rl
from sumo_rl.environment.observations import VANETObservationFunction
from sumo_rl.environment.attack_controller import global_orchestrator
from sumo_rl.environment.gan_attacker import init_gan_components

def train_adversarial_gan():
    print("======================================================")
    print("⚔️ ARÈNE ADVERSARIALE : ENTRAÎNEMENT DU cGAN")
    print("======================================================\n")

    # 1. Initialisation de l'environnement physique
    print("[*] Démarrage du moteur physique SUMO...")
    env = sumo_rl.SumoEnvironment(
        net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
        route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
        use_gui=False, # Pas de GUI pour entraîner l'IA rapidement
        num_seconds=5000, 
        delta_time=5,
        single_agent=True,
        observation_class=VANETObservationFunction,
        reward_fn='vanet'
    )

    # 2. Chargement du Défenseur (Le PPO est "Gelé", il ne s'entraîne plus)
    print("[*] Déploiement de l'Agent de Défense (PPO Frozen)...")
    try:
        defender_model = PPO.load("outputs/ppo_vanet_model")
    except Exception as e:
        print(f"[!] Erreur: Agent PPO introuvable. ({e})")
        return

    # 3. Initialisation du GAN (Attaquant + Jumeau Numérique)
    print("[*] Initialisation de l'Intelligence Hacking (Générateur & Discriminateur)...")
    state_dim = env.observation_space.shape[0]
    generator, discriminator, opt_G, opt_D, device = init_gan_components(state_dim)
    
    mse_loss = nn.MSELoss()
    
    num_episodes = 20
    print(f"\n[*] 🚀 DÉBUT DU JEU MIN-MAX ({num_episodes} Épisodes)...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        
        ep_gan_reward = 0
        ep_d_loss = 0
        ep_g_loss = 0
        
        while not done:
            # --- PHASE 1 : LE GAN PLANIFIE L'ATTAQUE ---
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device) # [1, state_dim]
            
            # Le Générateur crée l'arme (Vecteur de probabilités)
            attack_vector = generator(state_tensor)
            attack_array = attack_vector.cpu().detach().numpy()[0]
            
            # --- PHASE 2 : EXÉCUTION DANS LE MONDE PHYSIQUE ---
            # Le pont traduit le tenseur en attaque réelle sur le moteur
            ts_id = env.ts_ids[0]
            global_orchestrator.bridge_cGAN_tensor(ts_id, attack_array)
            
            # Le PPO subit l'attaque et prend une décision
            action, _ = defender_model.predict(obs, deterministic=True)
            next_obs, ppo_reward, terminated, truncated, _ = env.step(action)
            
            # La récompense du GAN est l'INVERSE de celle du PPO.
            # Plus le trafic bloque, plus le GAN est heureux.
            real_gan_reward = -ppo_reward
            ep_gan_reward += real_gan_reward
            
            # --- PHASE 3 : ENTRAÎNEMENT DU DISCRIMINATEUR (LE JUMEAU) ---
            # Il doit prédire la récompense exacte que le PPO vient de subir
            opt_D.zero_grad()
            pred_reward = discriminator(state_tensor, attack_vector.detach())
            target_reward = torch.FloatTensor([[real_gan_reward]]).to(device)
            
            loss_D = mse_loss(pred_reward, target_reward)
            loss_D.backward()
            opt_D.step()
            ep_d_loss += loss_D.item()
            
            # --- PHASE 4 : ENTRAÎNEMENT DU GÉNÉRATEUR (L'ATTAQUANT) ---
            # Il modifie ses poids pour que le Jumeau prédise une casse maximale
            opt_G.zero_grad()
            pred_reward_for_G = discriminator(state_tensor, attack_vector)
            
            # On veut maximiser pred_reward_for_G, donc on minimise son opposé
            loss_G = -pred_reward_for_G.mean()
            loss_G.backward()
            opt_G.step()
            ep_g_loss += loss_G.item()
            
            obs = next_obs
            done = terminated or truncated
            step += 1
            
        print(f"Épisode {episode+1}/{num_episodes} | Dégâts infligés au réseau : {ep_gan_reward:.2f} | Erreur Discriminateur: {ep_d_loss/step:.4f} | Loss GAN: {ep_g_loss/step:.4f}")

    env.close()
    
    # Sauvegarde des armes de destruction massive
    os.makedirs("outputs/gan", exist_ok=True)
    torch.save(generator.state_dict(), "outputs/gan/generator_model.pth")
    print("\n[✅] ENTRAÎNEMENT TERMINÉ. Le cerveau du GAN a été sauvegardé !")

if __name__ == "__main__":
    train_adversarial_gan()
