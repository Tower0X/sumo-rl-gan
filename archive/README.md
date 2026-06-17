# Archive

Ce dossier conserve des artefacts **historiques** qui ne font plus partie du
flux de travail actif, mais qui sont gardés pour la traçabilité scientifique
(reproductibilité du mémoire, comparaison avant/après).

## Contenu

- `legacy/train_marl_cooperative.py` — ancien script d'entraînement multi-agents
  qui vivait dans le dossier parasite `sumo-rl/` (avec un tiret), lequel
  masquait le vrai package `sumo_rl/`. L'approche multi-agents correcte
  (PettingZoo `parallel_env` + SuperSuit, parameter sharing) en a été extraite
  et réintégrée proprement à la racine du dépôt.

> Les fichiers d'archive ne sont **pas** maintenus et peuvent référencer des
> API obsolètes. Ne pas les importer depuis le code actif.
