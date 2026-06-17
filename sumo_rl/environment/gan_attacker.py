import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =====================================================================
# MODULE 3.1 : LE cGAN TEMPOREL (LSTM-based Adversarial Attacker)
# =====================================================================
# Évolution "Excellence" : 
# Contrairement au Perceptron simple (MLP), le LSTM possède une mémoire interne.
# Cela permet au Générateur de planifier des attaques graduelles et au 
# Discriminateur de détecter des anomalies sur une fenêtre temporelle.
# =====================================================================

class Generator(nn.Module):
    """
    L'IA HACKER RÉCURRENTE (LSTM Generator)
    Génère des séquences d'attaques intelligentes.
    """
    def __init__(self, state_dim, noise_dim=10, hidden_dim=128):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        
        # Le LSTM traite la séquence (État + Bruit)
        self.lstm = nn.LSTM(
            input_size=state_dim + noise_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # Couches de décision après la mémoire temporelle
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2)
        )
        
        # Sortie 1 : Probabilités des types d'attaques
        self.attack_type_layer = nn.Sequential(
            nn.Linear(64, 8), # Augmenté à 8 pour inclure les variantes DDoS/Sybil
            nn.Softmax(dim=-1)
        )
        
        # Sortie 2 : Intensité
        self.intensity_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state, hidden=None, noise=None):
        """
        state: (Batch, Seq, StateDim) ou (Batch, StateDim)
        """
        # Ajustement des dimensions si l'entrée n'a pas de dimension temporelle
        if state.dim() == 2:
            state = state.unsqueeze(1) # Ajout dim Seq=1
            
        if noise is None:
            noise = torch.randn(state.size(0), state.size(1), self.noise_dim, device=state.device)
        elif noise.dim() == 2:
            noise = noise.unsqueeze(1)
            
        x = torch.cat([state, noise], dim=-1)
        
        # Passage dans le LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # On traite le dernier élément de la séquence pour la décision
        features = self.feature_extractor(lstm_out[:, -1, :])
        
        attack_probs = self.attack_type_layer(features)
        intensity = self.intensity_layer(features)
        
        return torch.cat([attack_probs, intensity], dim=-1), hidden


class Discriminator(nn.Module):
    """
    LE JUMEAU NUMÉRIQUE RÉCURRENT (LSTM Surrogate Model)
    Imite SUMO et le PPO en analysant l'évolution temporelle.
    """
    def __init__(self, state_dim, attack_dim=9, hidden_dim=128): # 8 probas + 1 intensité
        super(Discriminator, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=state_dim + attack_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1) # Récompense prédite
        )

    def forward(self, state, attack_vector, hidden=None):
        if state.dim() == 2: state = state.unsqueeze(1)
        if attack_vector.dim() == 2: attack_vector = attack_vector.unsqueeze(1)
            
        x = torch.cat([state, attack_vector], dim=-1)
        lstm_out, hidden = self.lstm(x, hidden)
        
        predicted_reward = self.regression_head(lstm_out[:, -1, :])
        return predicted_reward, hidden

# =====================================================================
# INITIALISATION D'EXCELLENCE
# =====================================================================

# Chemin canonique du générateur LSTM entraîné (source de vérité unique).
DEFAULT_GENERATOR_PATH = "outputs/gan/generator_model_lstm.pth"


class GANLoadError(RuntimeError):
    """Raisée quand le générateur GAN ne peut pas être chargé correctement."""


def load_generator_strict(state_dim, path=DEFAULT_GENERATOR_PATH, device=None):
    """Charge le générateur LSTM de manière STRICTE.

    Contrairement à l'ancien ``try/except`` nu qui laissait l'attaquant tourner
    avec des poids ALÉATOIRES (rendant tout 'duel' factice), cette fonction échoue
    bruyamment:
    - ``GANLoadError`` si le fichier de poids est absent;
    - ``GANLoadError`` si l'architecture ne correspond pas (mismatch state_dict).

    L'appelant DOIT gérer cette exception et NE PAS poursuivre un duel sans GAN
    réellement entraîné.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path):
        raise GANLoadError(
            f"Modèle GAN introuvable: '{path}'. Entraînez le générateur avant de lancer "
            f"un duel adversarial (refus de tourner sur des poids aléatoires)."
        )
    generator = Generator(state_dim).to(device)
    try:
        state_dict = torch.load(path, map_location=device)
        generator.load_state_dict(state_dict)
    except Exception as exc:  # mismatch d'architecture, fichier corrompu, etc.
        raise GANLoadError(
            f"Échec du chargement du générateur depuis '{path}': {exc}. "
            f"Vérifiez que state_dim={state_dim} correspond au modèle entraîné."
        ) from exc
    generator.eval()
    return generator


def init_gan_components(state_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # On initialise avec une dimension d'attaque de 9 (8 types + 1 intensité)
    generator = Generator(state_dim).to(device)
    discriminator = Discriminator(state_dim, attack_dim=9).to(device)
    
    opt_G = optim.Adam(generator.parameters(), lr=5e-4) # LR plus petit pour LSTM
    opt_D = optim.Adam(discriminator.parameters(), lr=5e-4)
    
    return generator, discriminator, opt_G, opt_D, device
