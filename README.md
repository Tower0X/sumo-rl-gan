# Simulateur de Sécurité VANET (V2X-Resilience-Lab)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![SUMO](https://img.shields.io/badge/SUMO-1.18+-green.svg)
![RL](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)
![GAN](https://img.shields.io/badge/Adversarial-cGAN-red.svg)

## 📌 Présentation du Projet
Ce projet implémente une plateforme de **Co-Simulation (Trafic-Réseau)** avancée pour l'audit de résilience des infrastructures VANET (Vehicular Ad-hoc Networks). L'objectif est de concevoir un contrôleur de trafic intelligent capable de maintenir la sécurité routière (zéro collision) même en présence de cyber-attaques sophistiquées.

### Architecture Triple-Module
1.  **Module Défense (MARL-PPO) :** Un agent d'Apprentissage par Renforcement Multi-Agent entraîné avec une fonction de récompense "Fail-Safe". Il gère les feux de signalisation en anticipant les congestions et les freinages d'urgence.
2.  **Module Attaque (Orchestrateur) :** Un middleware capable d'injecter trois types de perturbations :
    *   **DoS Temporel :** Latence accrue sur les communications V2X.
    *   **Data Poisoning :** Corruption des données de télémétrie des capteurs.
    *   **Sybil Attack :** Simulation de faux véhicules pour saturer l'intersection.
3.  **Module Adversarial (cGAN) :** Une IA générative qui apprend à optimiser la virulence des attaques en fonction de l'état du trafic, créant un "jeu Min-Max" contre le défenseur PPO.

---

## 🛠️ Installation

### 1. Prérequis Système
*   **Eclipse SUMO :** [Télécharger et installer SUMO](https://eclipse.dev/sumo/).
*   **Variable d'environnement :** Assurez-vous que `SUMO_HOME` est correctement configuré dans votre système.

### 2. Installation de l'environnement Python
```bash
# Clonez le dépôt
git clone <repository_url>
cd sumo-rl-gan

# Créez un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate

# Installez les dépendances
pip install -r requirements.txt
# Ou via setup.py
pip install -e .
```

---

## 🚀 Utilisation

### Phase 1 : Entraînement du Défenseur (RL)
Pour entraîner l'agent PPO à gérer le trafic de manière sécurisée :
```bash
python train_vanet_ppo.py
```

### Phase 2 : Entraînement de l'Attaquant (cGAN)
Une fois le défenseur prêt, lancez l'entraînement adversarial pour tester sa résilience :
```bash
python train_gan_adversarial.py
```

### Phase 3 : Simulation de Bataille Finale
Pour visualiser l'affrontement entre le PPO et le GAN avec l'interface graphique SUMO :
```bash
python final_battle_gan.py
```

---

## 📊 Visualisation des Résultats
Les logs d'entraînement sont compatibles avec **TensorBoard** :
```bash
tensorboard --logdir outputs/marl_tensorboard
```

---

## 🏗️ Structure du Code
*   `sumo_rl/` : Fork personnalisé de l'environnement sumo-rl incluant les hooks de sécurité.
    *   `environment/traffic_signal.py` : Logique des récompenses VANET (Fail-Safe).
    *   `environment/attack_controller.py` : Orchestrateur des cyber-attaques.
    *   `environment/gan_attacker.py` : Cerveau du cGAN.
*   `assets/` : Diagrammes d'architecture et captures de simulation.
*   `experiments/` : Scénarios de test et scripts de baseline.

---

## 🎓 Équipe & Encadrement
*   **UE Projet :** INF4258 - GROUPE 10 (Université de Yaoundé 1)
*   **Membres :** MELONG LETHYCIA, DASSI MANDJO LEA JUSTINE, NGUEFACK TANGOMO CHRIS ARTHUR
*   **Sous la direction de :** Pr. Paulin MELATAGIA

---
*Ce projet a été développé dans le cadre d'un travail de recherche sur la résilience cyber-physique des systèmes de transport intelligents (ITS).*
