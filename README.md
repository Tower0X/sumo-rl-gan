# Simulateur de Sécurité VANET (V2X-Resilience-Lab)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![SUMO](https://img.shields.io/badge/SUMO-1.18+-green.svg)
![RL](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)
![Adversarial](https://img.shields.io/badge/Adversarial-Surrogate--Attacker-red.svg)

## 📌 Présentation du Projet
Ce projet implémente une plateforme de **Co-Simulation (Trafic-Réseau)** pour l'audit de résilience des infrastructures VANET (Vehicular Ad-hoc Networks). L'objectif est de concevoir un contrôleur de trafic par apprentissage par renforcement qui **mesure et minimise** les collisions et les freinages d'urgence, y compris lorsque les communications V2X sont dégradées par des cyber-attaques.

> **Note méthodologique.** La sécurité routière n'est pas postulée : les collisions sont **réellement mesurées** via SUMO (`--collision.action`) et la TraCI API, et pénalisées explicitement dans la récompense.

### Architecture Triple-Module
1.  **Module Défense (MARL récurrent) :** Les **16 feux** de la grille 4x4 sont des agents simultanés (API PettingZoo `parallel_env`) contrôlés par un **cerveau LSTM unique partagé** (RecurrentPPO + parameter sharing via SuperSuit). Un seul réseau apprend des expériences de tous les carrefours : apprentissage 16x plus dense, résilience distribuée. Il est entraîné avec une fonction de récompense **Fail-Safe à deux phases** (Eq. 3.4 / 3.5). La récompense commute sur l'état de communication `comm_ok` : en mode normal elle minimise pression + file ; en mode dégradé (perte V2X) elle ajoute un terme **quadratique** `−κ_quad·S²` sur la sur-file dangereuse `S = max(0, file − Q_safe)`. Les collisions et freinages d'urgence sont pénalisés dans les deux phases.
2.  **Module Attaque (Orchestrateur) :** Un middleware qui intercepte le vecteur d'observation et injecte **8 familles d'attaques** (`AttackType`) en ciblant exactement les bonnes features, sans jamais violer l'espace d'observation `Box[0,1]` :
    *   `TEMPORAL_DOS`, `FLOODING_DDOS`, `SLOWLORIS_DDOS` — perturbation de la latence V2X.
    *   `DATA_POISONING` — dissimulation des files d'attente (capteurs).
    *   `GHOST_VEHICLES`, `POSITION_JITTER` — attaques Sybil sur la densité.
    *   `JAMMER` — coupure totale de la communication (`comm_ok = 0`).
3.  **Module Adversarial (Attaquant par Jumeau Numérique, LSTM) :** Un attaquant récurrent (`SurrogateAdversarialAttacker`) produit des séquences d'attaques conditionnées par l'état du trafic. Il est entraîné contre un **jumeau numérique** (`SurrogateRewardModel`) qui imite la réponse du couple SUMO+PPO en prédisant la récompense sous attaque ; l'attaquant minimise cette récompense prédite par descente de gradient (Model-based Adversarial Attack). **Terminologie honnête :** ce n'est pas un GAN au sens strict (pas de jeu minimax réel/généré), et c'est nommé comme tel dans le code. Le chargement du modèle est **strict** : aucun duel n'est lancé avec des poids non entraînés.

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

## 🚀 Utilisation (pipeline complet)

Les phases s'enchaînent : on entraîne un défenseur, puis l'attaquant GAN contre ce défenseur, puis on évalue le duel.

### Phase 1 : Défenseur MARL coopératif (grille 4x4, 16 feux)
```bash
python train_marl_defender.py --timesteps 200000
# RecurrentPPO (LSTM) + SuperSuit parameter sharing
# -> outputs/recurrent_urban_shield_4x4.zip
```

### Phase 2 : Résilience APPRISE par curriculum adversarial (recommandé)
```bash
python train_marl_adversarial_curriculum.py \
    --phase1 60000 --phase2 100000 --phase3 120000
# Phase 1 (jitter) -> Phase 2 (attaques manuelles) -> Phase 3 (attaquant surrogate)
# -> outputs/recurrent_urban_shield_4x4.zip (défenseur durci)
```
> La résilience est **apprise** : le défenseur voit les attaques pendant l'entraînement (pas de zéro-shot).

### Phase 3 : Évaluation du duel (interface SUMO, 16 agents)
```bash
python evaluate_gan_vs_defender.py
```

### Dashboard de supervision
```bash
streamlit run app_dashboard.py   # ou, sous Windows : run_dashboard.bat
```

### Tests
```bash
pytest tests/
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
