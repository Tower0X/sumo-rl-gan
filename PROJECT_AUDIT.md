# Audit Technique Approfondi — Simulateur de Sécurité VANET (Groupe 10, INF4258)

> Scan complet du code source confronté au `Rapport_final_UE_Projet.pdf`.
> Objectif : transformer ce POC en une contribution de mémoire de Master de classe mondiale.
> Chaque constat est ancré sur une preuve `fichier:ligne`.

---

## 0. Synthèse exécutive

Le projet repose sur une idée scientifique forte et originale (un environnement de contrôle de trafic *network-aware* confronté à un attaquant neuronal). L'architecture en 5 modules est cohérente sur le papier. **Mais le code livré contient plusieurs erreurs structurelles qui invalident partiellement les affirmations du rapport** et qui, en l'état, ne tiendraient pas devant un jury exigeant.

Les trois plus graves :

1. **Le MARL 4x4 est un mensonge structurel dans les scripts GAN et de bataille** : `single_agent=True` ne pilote qu'**une seule intersection sur 16** (`env.py:281`, `env.py:313`). Le "Hive Mind de 16 RSU" n'existe pas dans `train_gan_recurrent.py` ni `final_battle_gan.py`.
2. **Le "cGAN" n'est pas un GAN** : le discriminateur est un régresseur de récompense (MSE), pas un classifieur réel/faux. Le gradient adversarial est court-circuité par un `detach()` numpy. C'est un *surrogate model*, ce qui est défendable — mais alors il faut cesser de l'appeler cGAN.
3. **Incohérence rapport↔code sur l'architecture du générateur** : le rapport LaTeX (`chap5_gan_adversarial.tex:31`) décrit un **MLP** ; le code de production (`gan_attacker.py:15`) est un **LSTM**. L'ancien artefact MLP (`generator_model.pth`, 5 sorties) était même **incompatible** avec le bridge actuel (9 sorties).

C'est exactement le symptôme que vous aviez repéré ("la mémoire des GAN était un MLP") : le projet a muté MLP→LSTM sans synchroniser le rapport, les artefacts et les dimensions.

---

## 1. ERREURS STRUCTURELLES CRITIQUES (à corriger avant soutenance)

### 1.1. 🔴 Le MARL 4x4 n'est pas réellement multi-agent dans 2 scripts clés
- `env.py:280-281` et `env.py:312-313` : en mode `single_agent=True`, l'environnement retourne **uniquement** `self.ts_ids[0]` pour l'observation, la récompense et n'applique l'action **que** sur le premier feu (`env.py:333-335`).
- Or `train_gan_recurrent.py:29` et `final_battle_gan.py:23` instancient la grille **4x4 (16 feux)** avec `single_agent=True`.
- **Conséquence** : pendant l'entraînement du GAN et la "bataille finale", **15 intersections sur 16 roulent en mode SUMO par défaut, non contrôlées**. Le défenseur ne voit qu'un seizième de la ville. La récompense `-16.98` du rapport (Scénario 2) et l'« immunité collective » ne sont pas démontrées par ces scripts.
- ✅ Le **vrai** MARL existe bien dans `marl_grid_boss.py:27` et `archives/duplicate_pkg_dir/train_marl_cooperative.py:25` via `parallel_env` + SuperSuit. C'est le bon chemin ; les scripts GAN doivent s'aligner dessus.

### 1.2. 🔴 Le générateur LSTM produit un argmax sur une dimension fantôme
- `gan_attacker.py:42` : `attack_type_layer` sort **8** probabilités.
- `attack_controller.py:7-15` : `AttackType` ne définit que **8 membres (index 0→7)**, mais l'index `0 = NONE`. Donc seuls 7 vrais types d'attaque existent.
- `attack_controller.py:71-76` : `bridge_cGAN_tensor` fait `argmax(probs[:8])` → index 0..7 → `AttackType(idx)`. Quand l'argmax tombe sur `0`, l'attaque est `NONE` (OK), mais le générateur a 8 logits softmax dont un seul correspond à "ne rien faire" — le déséquilibre n'est jamais documenté et fausse l'apprentissage adversarial (le générateur est récompensé pour des dégâts mais peut "gagner" en choisissant NONE).

### 1.3. 🔴 Violation systématique de l'espace d'observation (Box [0,1])
- `observations.py:90-93` : l'espace est borné `low=0, high=1`.
- Mais `attack_controller.py:115,119,124,153` : la corruption fait `corrupted_obs[-2] += noise` (jusqu'à `+10`) et le JAMMER force `corrupted_obs[-2] = 99.0` (`attack_controller.py:153`).
- **Conséquence** : les observations passées au PPO sortent de l'espace déclaré. SB3 ne plante pas, mais `VecNormalize`/clipping et la cohérence Box↔réseau sont violés ; tout wrapper `AssertOutOfBoundsWrapper` (utilisé côté PZ, `env.py:43`) lèverait une exception. La latence normalisée devrait être re-clampée après corruption.

### 1.4. 🔴 Le pipeline cGAN n'apprend pas un vrai jeu adversarial
- `train_gan_recurrent.py:71` : `attack_array = ...detach().numpy()` puis envoyé à SUMO. Le gradient **ne traverse jamais** l'environnement réel.
- Le discriminateur (`gan_attacker.py:79-108`) régresse `R_réel` par MSE (`train_gan_recurrent.py:94`). C'est un **surrogate model**, pas un discriminateur GAN (pas de distribution réel/faux, pas de `min_G max_D` classique).
- `train_gan_recurrent.py:95` : `loss_D.backward(retain_graph=True)` puis re-forward du discriminateur pour G (`:101`) — le générateur s'entraîne contre un jumeau qui n'a vu qu'**un seul échantillon** (pas de replay buffer), ce qui rend l'estimation du dommage extrêmement bruitée et non stationnaire.
- **Le rapport (`chap5:42-48`) décrit ce mécanisme honnêtement comme un surrogate**, mais le nomme "cGAN" partout. Il faut trancher la terminologie.

### 1.5. 🔴 Désynchronisation rapport ↔ code (risque de crédibilité en jury)
| Affirmation du rapport | Réalité du code |
|---|---|
| Générateur = MLP `[128,64]`, Softmax(4)+Sigmoid(1) (`chap5:31`, p.16 Tab 4.3) | Générateur = **LSTM** `hidden=128`, Softmax(**8**)+Sigmoid(1) (`gan_attacker.py:26,42,47`) |
| PPO = MLP `[64,64]` 20 000 timesteps (p.16) | `marl_grid_boss.py:49` = **RecurrentPPO MlpLstmPolicy**, 80 000 timesteps (`:65`) |
| "3 familles d'attaques" (résumé p.ii) | **7 types** implémentés (`attack_controller.py:9-15`) |
| 16 RSU partagent un réseau (p.10) | Vrai uniquement dans `marl_grid_boss.py`, faux dans les scripts GAN/battle |
| "0 collision physique dans les logs SUMO" (p.13) | **Aucun script ne lit `getCollidingVehiclesNumber()`** : la métrique-clé du mémoire n'est jamais mesurée dans le code |

### 1.6. 🔴 La métrique centrale du mémoire (collisions) n'est jamais mesurée
- Le titre, le résumé et le Scénario 3 reposent sur "**zéro collision**" (p.ii, p.13).
- Recherche dans tout le code actif : aucune lecture de `simulation.getCollidingVehiclesNumber()` ni `getCollisions()`. `_get_system_info` (`env.py:425-442`) ne collecte ni collisions ni freinages d'urgence.
- **C'est la faille scientifique la plus dangereuse** : l'affirmation phare n'est pas instrumentée.

---

## 2. INCOHÉRENCES FONCTIONNELLES (bugs réels mais non bloquants)

### 2.1. Télémétrie d'attaque jamais enregistrée
- `attack_controller.py:40-44` : `log_attack_impact()` est défini mais **jamais appelé**. Le commentaire "Enregistre la Télémétrie pour entraîner le futur cGAN" (`:41`) est trompeur. `outputs/attack_log.csv` ne contient donc que l'en-tête.

### 2.2. Orchestrateur global partagé = fuite d'état entre agents
- `attack_controller.py:158` : `global_orchestrator` est une **instance module-level unique**.
- `observations.py:83` : chaque feu appelle `global_orchestrator.corrupt_observation(self.ts, obs)`.
- En MARL 16 agents, `active_attacks` est indexé par `ts_id` (OK), mais le décrément du timer se fait dans `corrupt_observation` (`attack_controller.py:104`), appelé une fois par observation. Si un même feu est observé plusieurs fois par step (cas PZ AEC), le timer se décrémente trop vite. Couplage fort et difficile à tester unitairement.

### 2.3. `bridge_cGAN_tensor` reciblé arbitrairement
- `train_gan_recurrent.py:73` : `ts_id = env.ts_ids[step % len(env.ts_ids)]` — mais en `single_agent`, seul `ts_ids[0]` est observé/contrôlé. On attaque donc souvent des feux que le défenseur ne voit pas. L'attaque et la défense sont désalignées.

### 2.4. Hypothèses fragiles sur la structure du vecteur d'observation
- `attack_controller.py:133,141,146` : le Data Poisoning/Sybil suppose que "les queues sont dans la 2ᵉ moitié" via `mid = len(obs)//2`. Or l'observation réelle est `[phase_onehot, min_green, density..., queue..., latency, comm]` (`observations.py:78`). Le découpage `mid` ne correspond pas proprement à la frontière densité/queue dès que `num_green_phases ≠` au compte attendu. La corruption touche partiellement les mauvais indices.

### 2.5. Récompense "Fail-Safe duale" du rapport non implémentée telle quelle
- Le rapport (`chap?`, p.9, éq. 3.4/3.5) décrit une récompense à **deux phases** commutées par `C_ok` (nominal cinématique vs Fail-Safe quadratique).
- `traffic_signal.py:239-270` (`_vanet_reward`) : la récompense est une **somme pondérée fixe** (pression + anti-starvation + freinage + queue). **Il n'y a aucun branchement sur `comm_ok`**. Le "comportement Fail-Safe émergent" annoncé n'a pas de support dans la fonction de récompense livrée.

### 2.6. README pointe vers un script archivé
- `README.md:50` : `python train_vanet_ppo.py` — ce fichier est dans `archives/`. La commande échoue depuis la racine.
- `README.md:70` : `tensorboard --logdir outputs/marl_tensorboard` ignore `recurrent_marl_tensorboard/` (les vrais runs LSTM).

---

## 3. DETTE STRUCTURELLE & REPRODUCTIBILITÉ

### 3.1. Conflit de packaging
- `pyproject.toml:6` : le package s'appelle `sumo-rl` (l'upstream de Lucas Alegre). Le dossier local modifié est `sumo_rl/`. Un `pip install sumo-rl` écraserait/masquerait les modifications locales (attack hooks). Le projet doit **renommer son package** (ex. `vanet_sec_rl`) pour éviter toute collision silencieuse.

### 3.2. Artefacts incohérents (corrigé partiellement par cet audit)
- L'ancien `outputs/gan/generator_model.pth` (MLP, sorties `attack_type_layer (4,64)` — vérifié par inspection des poids) était **incompatible** avec le `Generator` LSTM (9 sorties). → **Archivé** dans `archives/deprecated_gan_mlp/generator_model_mlp.pth`.

### 3.3. Absence de notebooks de présentation
- Aucun `.ipynb` dans le dépôt. Pour un POC de mémoire, il manque les notebooks "récit" (architecture + courbes d'entraînement PPO/GAN + comparaison baseline). C'est un livrable attendu explicitement.

---

## 4. CE QUI A ÉTÉ FAIT DANS CET AUDIT (actions concrètes)

Fichiers déplacés vers `archives/` (compilation des fichiers actifs vérifiée, aucun import cassé) :
- `train_gan_adversarial.py` → `archives/deprecated_gan_mlp/` (pipeline MLP single-intersection remplacé par `train_gan_recurrent.py`).
- `outputs/gan/generator_model.pth` → `archives/deprecated_gan_mlp/generator_model_mlp.pth` (artefact MLP 5-sorties, incompatible).
- `test_env_4x4.py`, `test_ppo_recurrent.py` → `archives/dev_tests/` (scripts de smoke-test ad hoc).
- dossier doublon `sumo-rl/` (avec tiret) → contenu déplacé dans `archives/duplicate_pkg_dir/`, dossier supprimé (piège d'import vs `sumo_rl/`).

Racine désormais épurée : `app_dashboard.py`, `sim_runner.py`, `shared_state.py`, `marl_grid_boss.py`, `train_gan_recurrent.py`, `final_battle_gan.py`, `setup.py`.

---

## 5. FEUILLE DE ROUTE "CLASSE MONDIALE" (priorisée)

### Priorité 1 — Rétablir la vérité scientifique (bloquant soutenance)
1. **Aligner les scripts GAN/battle sur le vrai MARL** : remplacer `single_agent=True` par `parallel_env`+SuperSuit dans `train_gan_recurrent.py` et `final_battle_gan.py`, exactement comme `marl_grid_boss.py`. Le GAN doit attaquer et le défenseur doit voir les 16 feux.
2. **Instrumenter les collisions** : ajouter `getCollidingVehiclesNumber()` + comptage des freinages d'urgence (`a < -4.5`) dans `_get_system_info` (`env.py:425`), logguer par épisode. Sans ça, "0 collision" est invérifiable.
3. **Implémenter réellement la récompense Fail-Safe duale** dans `_vanet_reward` : brancher sur `self.comm_ok` (nominal cinématique vs quadratique dégradé), conformément à l'éq. 3.4/3.5 du rapport.
4. **Re-clamper l'observation après corruption** dans `corrupt_observation` (`np.clip(obs, 0, 1)` final) ou élargir l'`observation_space` à `high=+inf` sur les 2 derniers indices.

### Priorité 2 — Rendre le GAN défendable
5. **Trancher la terminologie** : soit assumer "**Surrogate-based Adversarial Attacker**" (honnête et original), soit implémenter un vrai cGAN (discriminateur réel/faux + génération conditionnée). Recommandation : assumer le surrogate, c'est la vraie contribution.
6. **Ajouter un replay buffer** pour l'entraînement du surrogate (sortir du régime 1-échantillon de `train_gan_recurrent.py:91`) et logguer `loss_D`, `loss_G`, dommage/épisode en CSV + TensorBoard.
7. **Activer `log_attack_impact()`** depuis `corrupt_observation`/`bridge_cGAN_tensor` pour constituer un dataset d'attaques exploitable hors-ligne.

### Priorité 3 — Reproductibilité & livrables
8. **Renommer le package** `sumo-rl` → `vanet_sec_rl` (`pyproject.toml:6`, `setup.py`) pour casser la collision avec l'upstream.
9. **Mettre à jour README** : commandes réelles (`marl_grid_boss.py`, `train_gan_recurrent.py`, `final_battle_gan.py`, `lancer_dashboard.bat`), bon `--logdir` TensorBoard.
10. **Synchroniser le rapport LaTeX** (`chap5:31`, Tab 4.2/4.3) : LSTM, RecurrentPPO, 7 types d'attaque, 80k timesteps.
11. **Produire 2 notebooks** : (a) *Architecture & Menaces* (diagrammes, flux défense/attaque, modèle de jitter), (b) *Entraînement & Évaluation* (courbes PPO, courbes surrogate/dommage, comparaison vs baseline SUMO fixe, scénarios 1/2/3 reproductibles avec métrique collisions).

---

## 6. Verdict

Le socle est réel et l'idée est publiable. Mais **trois affirmations centrales du mémoire (MARL 16 agents en duel, 0 collision mesurée, Fail-Safe émergent) ne sont pas soutenues par le code livré**. Les corriger (Priorité 1) transforme le projet d'un POC fragile en une démonstration rigoureuse et mémorable. La suite de ce travail (implémentation des P1→P3 et notebooks) est prise en charge.
