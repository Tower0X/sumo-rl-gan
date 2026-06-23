# Audit Technique Approfondi — Simulateur de Sécurité VANET (Groupe 10)

> Scan profond du dépôt `sumo-rl-gan` confronté au `Rapport_final_UE_Projet.pdf`.
> Objectif : relever les incohérences structurelles et scientifiques, comprendre l'existant,
> et dresser la feuille de route d'excellence pour la soutenance de Master.

---

## 0. Synthèse exécutive

Le projet repose sur une idée forte et originale (PPO défensif « Fail-Safe » vs attaquant
neuronal adversarial sur un VANET cyber-physique). L'architecture logicielle est crédible,
mais **plusieurs piliers annoncés dans le rapport ne sont pas réellement implémentés**, et
plusieurs bugs structurels invalident silencieusement les démonstrations. Sans correction,
le jury peut, en lisant le code, démonter les affirmations centrales du mémoire.

Gravité des problèmes (du plus critique au cosmétique) :

| # | Problème | Gravité | Impact soutenance |
|---|----------|---------|-------------------|
| C1 | La récompense « Fail-Safe à 2 phases » (Eq. 3.4/3.5) n'existe pas dans le code | 🔴 Critique | Contribution centrale non prouvée |
| C2 | « 0 collision » n'est jamais mesuré ; trivialement vrai par construction SUMO | 🔴 Critique | Résultat phare non étayé |
| C3 | L'attaquant GAN entraîné n'est jamais réellement utilisé (fallback poids aléatoires) | 🔴 Critique | Le « duel » est factice |
| C4 | Le PPO n'est jamais entraîné sous attaque (uniquement jitter) | 🟠 Majeur | « Résilience » = zéro-shot, pas apprise |
| C5 | Corruption d'observation par index heuristique → mauvaises features ciblées | 🟠 Majeur | Attaques ne font pas ce qui est annoncé |
| C6 | Violation des bornes de l'espace d'observation (valeurs 99.0, négatives) | 🟠 Majeur | Contrat Gym rompu, OOD non vu à l'entraînement |
| C7 | « cGAN » est en réalité un régresseur de récompense (surrogate), pas un GAN | 🟠 Majeur | Terminologie scientifique fausse |
| C8 | `final_battle` / `sim_runner` : `single_agent=True` sur grille 16 feux | 🟠 Majeur | « MARL 4x4 » ne contrôle qu'1 feu |
| C9 | Bug runtime dashboard : `env.traffic_signals` après wrap SuperSuit | 🟠 Majeur | Mode « Défense seule » plante |
| C10 | Télémétrie d'attaque (`log_attack_impact`) jamais appelée | 🟡 Moyen | Pas de dataset d'attaques |
| C11 | Dérive documentaire (README/rapport vs code réel) | 🟡 Moyen | Crédibilité / reproductibilité |
| C12 | Aucun notebook POC (exigence explicite du mémoire) | 🟡 Moyen | Livrable manquant |
| C13 | Duplication de packages `sumo-rl/` vs `sumo_rl/` ; fichiers morts | 🟢 Mineur | Encombrement |

---

## 1. Cartographie de l'existant (ce qui a été fait, et pourquoi)

### 1.1 Couche environnement (fork de `sumo-rl` de L. Alegre)
- `sumo_rl/environment/env.py` : wrapper Gymnasium + PettingZoo standard (quasi inchangé
  vs upstream). Ajout : rien de spécifique VANET ici.
- `sumo_rl/environment/traffic_signal.py` : ajout du flag `self.comm_ok = True` (ligne 93)
  et de la reward `_vanet_reward` (lignes 239-270) enregistrée sous la clé `"vanet"`.
- `sumo_rl/environment/observations.py` : ajout de `VANETObservationFunction` qui :
  1. concatène `[phase_onehot, min_green, density, queue, latence_normalisée, comm_flag]`,
  2. simule un jitter gaussien sur la latence (μ=10 ms, σ=5 ms),
  3. appelle `global_orchestrator.corrupt_observation(ts, obs)` pour empoisonner l'état.
- `sumo_rl/environment/attack_controller.py` : `CyberPhysicalAttackOrchestrator`,
  middleware Man-In-The-Middle, 8 types d'attaques (`AttackType`), pilotable manuellement
  ou via tenseur GAN (`bridge_cGAN_tensor`).
- `sumo_rl/environment/gan_attacker.py` : `Generator` + `Discriminator` **LSTM**
  (la migration MLP→LSTM annoncée a bien été faite ici).

### 1.2 Scripts d'entraînement
- `archives/train_vanet_ppo.py` : PPO MLP, intersection simple, 20 000 pas, reward `vanet`.
- `archives/train_marl_cooperative.py` : MARL PPO (PettingZoo + SuperSuit).
- `marl_grid_boss.py` : `RecurrentPPO` (LSTM) + Parameter Sharing sur 4x4, 80 000 pas.
- `archives/.../train_gan_adversarial.py` : GAN MLP (legacy), sauvegarde `generator_model.pth`.
- `train_gan_recurrent.py` : GAN LSTM, sauvegarde `generator_model_lstm.pth`.

### 1.3 Runtime / démonstration
- `final_battle_gan.py` : duel PPO vs GAN avec GUI SUMO.
- `app_dashboard.py` + `sim_runner.py` + `shared_state.py` : dashboard Streamlit « SOC »
  (cartes télémétrie, courbes, logs, modes Défense / Attaque manuelle / Duel cGAN).

---

## 2. Incohérences critiques (analyse détaillée)

### C1 — Le « Fail-Safe à deux phases » N'EST PAS implémenté 🔴
**Annoncé (rapport §3.2.4, Eq. 3.4 & 3.5).** La récompense est censée commuter selon `C_ok` :
- Phase nominale (`C_ok=1`) : pénalité cinématique, **−5.0** par freinage d'urgence (< −4.5 m/s²).
- Phase dégradée (`C_ok=0`) : `R = −Σ (Q_i · λ)²` (quadratique) → comportement « horloge suisse »
  émergent, présenté comme LE résultat scientifique du mémoire.

**Réel (`traffic_signal.py:239-270`).** `_vanet_reward` = `2·pressure + 2·anti_starvation +
0.5·braking + 0.5·queue`, où le freinage ne pénalise que **−0.1** (et non −5.0), et **aucune
ligne ne lit `self.comm_ok`**. Vérification par recherche globale : `comm_ok` n'apparaît que
dans `attack_controller.py` (écriture) et `observations.py` (lecture comme feature). Il n'y a
**aucune commutation de phase de récompense**, **aucun terme quadratique** Eq. 3.5.

**Conséquence.** Le « comportement Fail-Safe émergent » ne peut pas être produit par ce code.
C'est l'équivalent exact du « la mémoire du GAN était un MLP » : l'affirmation phare du
rapport n'a pas de support dans l'implémentation.

### C2 — « Zéro collision » n'est jamais mesuré 🔴
- `num_seconds`, `time_to_teleport=-1` (pas de téléportation), modèle de poursuite de Krauss :
  SUMO **empêche les collisions par construction** sur le modèle par défaut (pas de
  `--collision.action`, pas de jonction `model`). « 0 collision » est donc **trivialement vrai**
  et ne démontre rien sur la résilience.
- Aucun script ne lit `simulation.getCollidingVehiclesNumber()` ni n'active
  `--collision.action warn/teleport`. Le résultat phare n'a **aucun artefact** qui l'étaye.

### C3 — L'attaquant GAN entraîné n'est jamais réellement chargé 🔴
- `final_battle_gan.py:46` et `sim_runner.py:73` chargent `generator_model_lstm.pth` dans un
  `try/except` **nu**. En cas d'échec (fichier absent, mismatch de dimensions d'état entre
  l'env d'entraînement et celui de démo), l'exception est avalée et **le Generator reste à poids
  aléatoires**. Le « duel d'excellence » oppose alors le PPO à un **attaquant non entraîné**.
- Incohérence de nommage historique : `train_gan_adversarial.py` sauvegarde
  `generator_model.pth`, mais la démo charge `generator_model_lstm.pth`.

### C4 — Le PPO n'est jamais entraîné sous attaque 🟠
- `train_vanet_ppo.py` instancie `VANETObservationFunction` (jitter OK) mais **aucune attaque
  n'est déclenchée pendant l'entraînement** (`global_orchestrator.active_attacks` reste vide).
- Donc la « résilience » observée en test est **zéro-shot** : l'agent n'a jamais vu de DoS /
  Poisoning / Sybil à l'entraînement. À encadrer honnêtement, ou à corriger par un
  entraînement adversarial / curriculum (recommandé).

### C5 — Corruption d'observation par index heuristique fragile 🟠
- Layout réel : `[phase(P), min_green(1), density(L), queue(L), latence(1), comm(1)]`,
  longueur `P+2+2L`.
- `attack_controller.corrupt_observation` utilise `mid = len(obs)//2` puis cible
  `obs[mid:-2]` (Data Poisoning) ou `obs[1:mid]` (Ghost/Sybil). Ce `mid` ne coïncide avec la
  frontière density/queue **que par accident**. En général, le poisoning frappe un **mélange**
  de phase/min_green/density/queue → l'attaque **ne fait pas ce que le rapport annonce**
  (`Q̃ = Q·(1−I)`, `D̃ = min(1, D+I)`). Il faut indexer via les vraies frontières
  (`num_green_phases`, `len(lanes)`) exposées par le `TrafficSignal`.

### C6 — Violation des bornes de l'espace d'observation 🟠
- L'espace est `Box(low=0, high=1)`. Or :
  - `JAMMER` écrit `obs[-2]=99.0` ;
  - `TEMPORAL_DOS/FLOODING/SLOWLORIS` ajoutent un bruit pouvant dépasser 1.0 ;
  - `TEMPORAL_DOS` peut produire des valeurs négatives.
- SB3 ne clippe pas les observations `Box` : le contrat est rompu, et surtout l'agent
  rencontre des valeurs **hors distribution jamais vues à l'entraînement** (cf. C4).

### C7 — « cGAN » mal nommé : c'est un attaquant par modèle-surrogate 🟠
- Le `Discriminator` est entraîné en **régression MSE** vers `−R_ppo` (jumeau différentiable),
  **pas** en classification réel/généré, sans critique de Wasserstein, sans label conditionnel
  adversarial. Le `Generator` maximise le dommage prédit par ce surrogate.
- C'est une **attaque par gradient via modèle-surrogate** (élégante et défendable !), mais
  **ce n'est pas un cGAN** au sens de Goodfellow/Mirza. Soit corriger la terminologie du
  rapport, soit implémenter un vrai critique (WGAN-GP) — décision à prendre.
- De plus : surrogate appris en SGD 1-échantillon/pas, sans replay buffer → cible très bruitée.

### C8 — `single_agent=True` sur une grille de 16 feux 🟠
- `final_battle_gan.py:23` met `single_agent=True` sur le réseau 4x4 (16 feux). En mode
  single-agent, seul `ts_ids[0]` est piloté ; les 15 autres restent en programme par défaut.
  Le « duel sur grille urbaine 4x4 / MARL » ne contrôle donc **qu'une seule intersection**.

### C9 — Bug runtime du dashboard (mode Défense seule) 🟠
- `sim_runner.py:97` : `for tid in ts_ids: env.traffic_signals[tid].comm_ok = True`. Mais à ce
  stade `env` a été wrappé par SuperSuit (`pettingzoo_env_to_vec_env_v1` →
  `concat_vec_envs_v1`) et **n'expose plus `.traffic_signals`** → `AttributeError` au premier
  pas en mode « Défense seule ».

### C10 — Télémétrie d'attaque morte 🟡
- `log_attack_impact()` (attack_controller) est implémentée mais **jamais appelée**. Le
  commentaire « Enregistre la télémétrie pour entraîner le futur cGAN » est donc trompeur.
  Aucun dataset d'attaques exploitable n'est constitué.

### C11 — Dérive documentaire 🟡
- `README.md` lance `python train_vanet_ppo.py` (déplacé dans `archives/`).
- Rapport Table 4.3 : GAN MLP `Softmax(4)` — le code actuel est LSTM `Softmax(8)`.
- Rapport : « Discriminateur = MLP » — code = LSTM. Plusieurs hyperparamètres divergent.

### C12 — Aucun notebook POC 🟡
- Exigence explicite (mémoire POC). Aucun `.ipynb` dans le dépôt. Il faut 1–2 notebooks
  « ultra-détaillés » : architecture + entraînement + courbes + confrontation.

### C13 — Encombrement / duplication 🟢
- Dossier parasite `sumo-rl/` (avec tiret) contenant un unique `train_marl_cooperative.py`,
  doublon conceptuel de `sumo_rl/`.
- Fichiers utilitaires morts : `outputs/plot.py`, `sumo_rl/nets/4x4-Lucas/metrics/result_plot.py`.
- Binaires lourds versionnés (`.pth`, `.zip`, `tfevents`) sans git-lfs.

---

## 3. Bugs ponctuels supplémentaires
- `sim_runner.py:143` : latence affichée = `mean(obs[:,-2])·10`. Or `obs[-2]` est déjà
  `latence/0.1` ; l'unité affichée (« ms ») est incohérente (~1 au lieu de ~10 ms).
- `sim_runner.py:122` : `episode_start` de forme `(1,)` passé à un VecEnv de 16 agents
  concaténés (devrait être `(16,)`) → gestion d'état LSTM fragile.
- `app_dashboard.py:223-231` : seuils de récompense (« Fail-Safe Actif » si < −5) purement
  cosmétiques, non reliés à `comm_ok` ni à l'échelle réelle de `_vanet_reward`.

---

## 4. Forces réelles à conserver et valoriser
- Idée de recherche solide et différenciante (network-aware RL + adversaire neuronal).
- Migration LSTM (Generator/Discriminator récurrents) propre et bien dimensionnée.
- Orchestrateur d'attaques riche (8 familles, MITM, pilotage manuel + GAN).
- Dashboard Streamlit de grande qualité visuelle (atout fort en soutenance).
- Architecture MARL (PettingZoo + SuperSuit + Parameter Sharing) en place pour le 4x4.

---

## 5. Feuille de route d'excellence (priorisée)

### Lot A — Vérité scientifique (indispensable avant soutenance)
1. **Implémenter réellement la reward Fail-Safe à 2 phases** pilotée par `comm_ok`
   (Eq. 3.4/3.5), avec terme quadratique en mode dégradé et −5.0 freinage en nominal.
2. **Mesurer les collisions** : activer `--collision.action warn` + lire
   `getCollidingVehiclesNumber()` et journaliser par épisode (artefact CSV).
3. **Entraîner le PPO sous attaques** (curriculum / domain randomization) pour que la
   résilience soit *apprise* et non zéro-shot.
4. **Corriger l'indexation des attaques** (frontières réelles density/queue) + **borner**
   les observations corrompues dans `[0,1]` (sauf canal latence/`comm` documenté).
5. **Brancher réellement le GAN entraîné** (supprimer les `except` nus, vérifier les
   dimensions, unifier le nom d'artefact) et corriger `single_agent` → MARL réel en démo.
6. **Trancher la terminologie GAN** : soit renommer « attaque par modèle-surrogate
   différentiable », soit implémenter un vrai critique WGAN-GP.

### Lot B — Reproductibilité & instrumentation
7. Activer `log_attack_impact()` → dataset d'attaques + analyses offline.
8. Logging unifié (CSV + TensorBoard) : G-loss, D-loss, reward défenseur, taux de succès
   d'attaque, collisions, file d'attente.
9. Pipeline GAN unique et nommage d'artefacts normalisé.

### Lot C — Livrables POC (notebooks)
10. **Notebook 1 — Architecture & Méthodologie** : schémas, MDP, modèle réseau, arsenal
    d'attaques, surrogate, MARL.
11. **Notebook 2 — Entraînement & Résultats** : courbes PPO, duel GAN, scénarios 1/2/3,
    comparaison baseline statique vs PPO, collisions, Fail-Safe.

### Lot D — Hygiène du dépôt
12. Consolider tous les fichiers morts dans `archives/` (déjà initié) + supprimer `sumo-rl/`.
13. Mettre à jour `README.md` et `docs/` pour refléter les scripts réellement utilisables.
14. `.gitignore` + git-lfs pour les binaires lourds.

---

## 6. Note sur l'exécution
Les lots A/B impliquent des **ré-entraînements SUMO + SB3/PyTorch** qui doivent tourner sur
la machine de l'équipe (SUMO_HOME, GPU). Les corrections de code (logique reward, indexation,
branchement GAN, dashboard, logging) sont applicables et vérifiables hors entraînement lourd.
