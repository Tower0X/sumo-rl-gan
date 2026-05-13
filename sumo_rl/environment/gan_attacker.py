import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =====================================================================
# MODULE 3 : LE cGAN CYBER-PHYSIQUE (Adversarial Attacker)
# =====================================================================
# Architecture d'Excellence : 
# SUMO étant un simulateur physique, il n'est pas "différentiable" 
# (on ne peut pas faire de backpropagation directe à travers lui).
# Pour contourner cela, nous utilisons un véritable cGAN :
# 1. Le DISCRIMINATEUR agit comme un "Jumeau Numérique" de SUMO. Il
#    apprend à prédire la récompense du PPO en fonction de l'attaque.
# 2. Le GÉNÉRATEUR crée les attaques et s'entraîne en faisant de la 
#    backpropagation à travers le Discriminateur !
# =====================================================================

class Generator(nn.Module):
    """
    L'IA HACKER (Le Générateur)
    Observe le trafic et génère un vecteur d'attaque.
    """
    def __init__(self, state_dim, noise_dim=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        # Le réseau prend en entrée l'état de l'intersection + un bruit aléatoire
        self.net = nn.Sequential(
            nn.Linear(state_dim + noise_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
        )
        
        # Sortie 1 : Type d'attaque (3 types + 1 option "Ne rien faire")
        # Softmax pour obtenir des probabilités
        self.attack_type_layer = nn.Sequential(
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )
        
        # Sortie 2 : Intensité de l'attaque (entre 0.0 et 1.0)
        self.intensity_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state, noise=None):
        if noise is None:
            noise = torch.randn(state.size(0), self.noise_dim, device=state.device)
            
        x = torch.cat([state, noise], dim=-1)
        features = self.net(x)
        
        attack_probs = self.attack_type_layer(features)
        intensity = self.intensity_layer(features)
        
        # On concatène les probabilités et l'intensité pour former le "Vecteur d'Attaque"
        return torch.cat([attack_probs, intensity], dim=-1)


class Discriminator(nn.Module):
    """
    LE JUMEAU NUMÉRIQUE (Le Discriminateur / Surrogate Model)
    Il tente d'imiter le moteur SUMO et le PPO. Il prend un État + une Attaque,
    et tente de prédire la Récompense (ou plutôt, la Pénalité) que le PPO subira.
    """
    def __init__(self, state_dim, attack_dim=5): # 4 probas + 1 intensité
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + attack_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1) # Prédit un score scalaire (la récompense)
        )

    def forward(self, state, attack_vector):
        x = torch.cat([state, attack_vector], dim=-1)
        predicted_reward = self.net(x)
        return predicted_reward

# =====================================================================
# FONCTIONS UTILITAIRES POUR L'ENTRAÎNEMENT DU GAN
# =====================================================================

def init_gan_components(state_dim):
    """ Initialise les réseaux et les optimiseurs """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = Generator(state_dim).to(device)
    discriminator = Discriminator(state_dim).to(device)
    
    # On utilise l'optimiseur Adam, standard pour les GANs
    opt_G = optim.Adam(generator.parameters(), lr=1e-3)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-3)
    
    return generator, discriminator, opt_G, opt_D, device
