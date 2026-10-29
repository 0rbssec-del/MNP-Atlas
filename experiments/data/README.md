# Données externes

`akarlar_selected_prompts.json` : les 8 prompts « les plus équilibrés » utilisés
par Akarlar (2026) pour son patching (indices 20, 24, 10, 16, 22, 25, 27, 29),
avec ses listes `ground_truth` et `wrong_indicators`. Extraits sans exécution
(analyse `ast`) de `experiments/e05_find_hallucination/multitoken_hallucination.py`
du dépôt https://github.com/akarlaraytu/trajectory-commitment, sélection
recalculée depuis `bifurcation_Qwen_Qwen2.5-1.5B.json` (tri par min(C, H)).

Copyright (c) 2026 Gokturk Aytug Akarlar / Chimera Research Initiative,
licence MIT.
