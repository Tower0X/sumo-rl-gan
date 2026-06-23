import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =====================================================================
# MODULE 3.1 : ATTAQUANT ADVERSARIAL PAR JUMEAU NUMERIQUE (LSTM)
# =====================================================================
# HONNETETE TERMINOLOGIQUE (important pour la soutenance) :
#
#   Ce module N'EST PAS un GAN au sens strict (il n'y a pas de jeu
#   minimax generateur/discriminateur ou le discriminateur classe
#   'reel vs genere'). C'est un ATTAQUANT ADVERSARIAL ENTRAINE PAR
#   JUMEAU NUMERIQUE (surrogate model) :
#
#     - Le "Generateur" (Attacker) est un reseau LSTM qui produit une
#       sequence d'attaques conditionnee par l'etat du trafic.
#     - Le "Surrogate" est un modele de regression LSTM qui IMITE la
#       reponse du couple SUMO+PPO : il predit la recompense que le
#       defenseur obtiendrait sous une attaque donnee.
#     - L'attaquant est entraine pour MINIMISER la recompense predite
#       par le surrogate (objectif: maximiser les degats), via une
#       descente de gradient differentiable a travers le surrogate.
#
#   C'est une methode valide et defendable (Model-based Adversarial
#   Attack / surrogate gradient attack). On la nomme correctement
#   partout pour qu'aucun jury ne puisse nous prendre en defaut sur
#   la terminologie.
#
#   Apport "LSTM" : contrairement a un MLP, la memoire interne permet
#   a l'attaquant de planifier des attaques GRADUELLES (sequentielles)
#   et au surrogate de modeliser des dynamiques temporelles.
# =====================================================================


class SurrogateAdversarialAttacker(nn.Module):
    """Attaquant adversarial recurrent (LSTM).

    Genere une sequence d'attaques intelligentes conditionnees par
    l'etat du trafic. Entraine pour minimiser la recompense predite
    par le jumeau numerique (``SurrogateRewardModel``).

    Sortie : ``[attack_type_probs (8), intensity (1)]`` = 9 dimensions.
    """

    def __init__(self, state_dim, noise_dim=10, hidden_dim=128):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim

        # Le LSTM traite la sequence (Etat + Bruit)
        self.lstm = nn.LSTM(
            input_size=state_dim + noise_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Couches de decision apres la memoire temporelle
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
        )

        # Sortie 1 : Probabilites des types d'attaques (8 familles)
        self.attack_type_layer = nn.Sequential(
            nn.Linear(64, 8),
            nn.Softmax(dim=-1),
        )

        # Sortie 2 : Intensite [0, 1]
        self.intensity_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, hidden=None, noise=None):
        """state: (Batch, Seq, StateDim) ou (Batch, StateDim)."""
        # Ajustement des dimensions si l'entree n'a pas de dimension temporelle
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Ajout dim Seq=1

        if noise is None:
            noise = torch.randn(state.size(0), state.size(1), self.noise_dim, device=state.device)
        elif noise.dim() == 2:
            noise = noise.unsqueeze(1)

        x = torch.cat([state, noise], dim=-1)

        # Passage dans le LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # On traite le dernier element de la sequence pour la decision
        features = self.feature_extractor(lstm_out[:, -1, :])

        attack_probs = self.attack_type_layer(features)
        intensity = self.intensity_layer(features)

        return torch.cat([attack_probs, intensity], dim=-1), hidden


class SurrogateRewardModel(nn.Module):
    """Jumeau numerique recurrent (LSTM surrogate model).

    Imite la reponse du couple SUMO + PPO en analysant l'evolution
    temporelle : etant donne un etat et un vecteur d'attaque, il PREDIT
    la recompense que le defenseur obtiendrait. Ce n'est PAS un
    discriminateur GAN (pas de classification reel/genere) mais un
    regresseur de recompense, utilise comme proxy differentiable pour
    entrainer l'attaquant.
    """

    def __init__(self, state_dim, attack_dim=9, hidden_dim=128):  # 8 probas + 1 intensite
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=state_dim + attack_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),  # Recompense predite
        )

    def forward(self, state, attack_vector, hidden=None):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if attack_vector.dim() == 2:
            attack_vector = attack_vector.unsqueeze(1)

        x = torch.cat([state, attack_vector], dim=-1)
        lstm_out, hidden = self.lstm(x, hidden)

        predicted_reward = self.regression_head(lstm_out[:, -1, :])
        return predicted_reward, hidden


# =====================================================================
# CHARGEMENT STRICT (jamais de poids aleatoires)
# =====================================================================

# Chemin canonique de l'attaquant LSTM entraine (source de verite unique).
DEFAULT_GENERATOR_PATH = "outputs/gan/generator_model_lstm.pth"


class GANLoadError(RuntimeError):
    """Raisee quand l'attaquant ne peut pas etre charge correctement."""


def load_generator_strict(state_dim, path=DEFAULT_GENERATOR_PATH, device=None):
    """Charge l'attaquant LSTM de maniere STRICTE.

    Contrairement a l'ancien ``try/except`` nu qui laissait l'attaquant
    tourner avec des poids ALEATOIRES (rendant tout 'duel' factice),
    cette fonction echoue bruyamment :
    - ``GANLoadError`` si le fichier de poids est absent ;
    - ``GANLoadError`` si l'architecture ne correspond pas (mismatch state_dict).

    L'appelant DOIT gerer cette exception et NE PAS poursuivre un duel
    sans attaquant reellement entraine.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path):
        raise GANLoadError(
            f"Modele attaquant introuvable: '{path}'. Entrainez l'attaquant avant de "
            f"lancer un duel adversarial (refus de tourner sur des poids aleatoires)."
        )
    generator = SurrogateAdversarialAttacker(state_dim).to(device)
    try:
        state_dict = torch.load(path, map_location=device)
        generator.load_state_dict(state_dict)
    except Exception as exc:  # mismatch d'architecture, fichier corrompu, etc.
        raise GANLoadError(
            f"Echec du chargement de l'attaquant depuis '{path}': {exc}. "
            f"Verifiez que state_dim={state_dim} correspond au modele entraine."
        ) from exc
    generator.eval()
    return generator


def init_gan_components(state_dim, lr=5e-4):
    """Initialise l'attaquant et son jumeau numerique pour l'entrainement.

    Returns:
        attacker, surrogate, opt_attacker, opt_surrogate, device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dimension d'attaque de 9 (8 types + 1 intensite)
    attacker = SurrogateAdversarialAttacker(state_dim).to(device)
    surrogate = SurrogateRewardModel(state_dim, attack_dim=9).to(device)

    opt_attacker = optim.Adam(attacker.parameters(), lr=lr)  # LR plus petit pour LSTM
    opt_surrogate = optim.Adam(surrogate.parameters(), lr=lr)

    return attacker, surrogate, opt_attacker, opt_surrogate, device


# =====================================================================
# COMPATIBILITE ASCENDANTE
# =====================================================================
# Les anciens noms (Generator / Discriminator) restent importables pour
# ne casser aucun script existant, mais sont des ALIAS clairement
# documentes vers la terminologie correcte ci-dessus.
Generator = SurrogateAdversarialAttacker
Discriminator = SurrogateRewardModel
