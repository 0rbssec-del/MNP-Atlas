# Moteur v0.2 : ce qui change et comment s'en servir pour l'atlas

`trajectory_bifurcation_core.py` v0.2.0 aligne le moteur sur le protocole des notebooks SGZ (A1c–B2) et ajoute les
analyses dont l'atlas a besoin : régimes, bifurcation par préfixe, valeur ajoutée sur les jetons, choc. L'API v0.1.x
reste disponible ; les deux expériences du dépôt fixent explicitement leurs anciens réglages et restent reproductibles.

## 1. Ce qui change

| | v0.1.5 | v0.2.0 |
|---|---|---|
| Échantillonneur | `torch.multinomial` + graine | **Gumbel à graine explicite par ligne et par position** (`sampler="gumbel"`) ; `"multinomial"` disponible |
| Température, top-p | 0,8 ; 0,95 | **0,6 ; 1,0** (SGZ) |
| Prompt | gabarit de chat toujours | **brut pour les modèles « -Base »**, chat sinon (`chat=` pour forcer) |
| Jetons spéciaux | arrêt sur EOS | **interdits** avec Gumbel (longueur fixe, SGZ) ; `forbid_special=False` pour l'arrêt sur EOS |
| Précision GPU | bf16 | **fp16** si le GPU n'a pas de bf16 natif (T4), bf16 sinon |
| Génération | une trajectoire à la fois | **par lots** (`generer_ensemble`, remplissage à gauche, KV-cache) |
| Entropie | Shannon | Shannon **et Tsallis q = 1,45** (loi du modèle, T = 1) |
| Paires de patching | négatif, positif, informative | + **`pair_detail`** : `informative_context` (dernier jeton identique) / `informative_token` |
| Recovery | rapport toujours calculé | **None si le dénominateur est trop petit** (JS < 0,05 nat, Δ log-vraisemblance < 0,5 nat) |
| Divergences | JS, distances | + **`corr_js_divergence_jetons`** (critère C1) et JS des seules paires divergentes |

Pourquoi Gumbel : avec une graine par (ligne, position), deux branches qui partagent un préfixe reçoivent le même
bruit à chaque position (nombres aléatoires communs). Un témoin négatif est alors identique à l'arrondi près des logits,
au lieu de dépendre de la position d'un seuil cumulé dans `torch.multinomial`. (Le témoin négatif à recouvrement 0 de
`resultats_multi_prompt.json` n'en est pas un exemple : ses jetons ne divergent pas, `first_divergence_step` = None ;
le 0 venait du Jaccard d'une continuation vide, fin de texte immédiate, corrigé dans `bdec38e`.)

## 2. Démarrage

```python
from trajectory_bifurcation_core import TrajectoryBifurcationEngine
eng = TrajectoryBifurcationEngine.sgz("Qwen/Qwen3-1.7B-Base", max_new_tokens=32, seed=20261029)
ens = eng.generer_ensemble(prompts, n=192, n_steps=32, layers=[17], keep_steps=True, mean_layers=[17])
```

`Ensemble` : `tokens (N, T)`, `entropy_shannon`, `entropy_tsallis (N, T)`, `hidden_steps[l] (N, T, d)` (float16),
`hidden_mean[l] (N, d)`, `prompt_of`, `row_ids`. Lignes d'un prompt : `ens.lignes_du_prompt(p)`.

Conventions (identiques à la v0.1.x) : `hidden_steps[l][i, k]` est l'état de la couche l qui produit la loi du jeton k
(lecture du jeton k − 1) ; `entropy_*[i, k]` est l'entropie de cette loi. **Couche l du moteur = `hidden_states[l + 1]`
de Hugging Face** (sortie brute de la couche, avant la norme finale). La couche principale de B2, `hidden_states[⌊2L/3⌋]`,
est donc `l = ⌊2L/3⌋ − 1` (17 pour Qwen3-1.7B, 23 pour SmolLM3-3B).

## 3. Les quatre contraintes de l'atlas (résultats SGZ B2 et réplication d'Akarlar)

1. **Carte sur les états cachés, pas sur l'entropie.** Dans B2, les profils d'entropie (32 positions) ne distinguent
   même pas deux prompts différents dans plus de la moitié des paires ; les états cachés y arrivent à 100 %.
2. **Des régimes pour une minorité de prompts.** B2 : bassins pour 12 à 19 % des prompts à T = 0,6, presque aucun à
   T = 1,0. `analyser_regimes` peut répondre « CONTINUUM » ou « MIXTE » : ce sont des résultats, pas des échecs.
3. **Pas de jeton de bascule unique.** Le premier jeton engage à peine l'issue (réplication d'Akarlar) et le régime
   est rarement fixé par un préfixe (B2 : gain du préfixe sur le hasard ≈ 0,17 en médiane). `bifurcation_prefixe`
   rapporte le gain, pas seulement un k*.
4. **Valeur ajoutée sur les jetons.** Une carte n'apporte quelque chose que si l'état caché à t prédit l'issue mieux
   qu'un lecteur de jetons du même préfixe : `valeur_ajoutee_sur_jetons` (AUC hors échantillon, écart et IC95, verdict).

Une « bifurcation » JS (`detect_bifurcation`) suit presque toujours la divergence des jetons
(`corr_js_divergence_jetons` ≥ 0,92 dans les 16 runs de v0.1.5) : elle situe où les trajectoires se séparent, pas une
dynamique interne.

## 4. Analyses ajoutées

- `analyser_regimes(R, seed)` : `R[p] (n_p, d)` par prompt. Test de bimodalité par moitiés indépendantes (2-moyennes sur
  A, dip de Hartigan sur B), substitut gaussien (calibration), mélanges de paires de prompts (puissance),
  Benjamini-Hochberg ; verdict BASSINS / MIXTE / CONTINUUM / NON INTERPRÉTABLE. Code de B2, mêmes graines.
- `etiquettes_regimes(X)` : deux régimes par 2-moyennes (pour les prompts multimodaux).
- `bifurcation_prefixe(TK, lab, S)` : premier k où le préfixe prédit le régime (≥ 0,9, ≥ 0,25 au-dessus du hasard),
  gain maximal, entropie à k*.
- `valeur_ajoutee_sur_jetons(H_t, prefixes, y, groupes)` : sondes logistiques sur l'état et sur un sac de jetons haché
  (plus l'identité des 3 derniers jetons). VALEUR AJOUTÉE si l'écart d'AUC a une borne basse > 0,05.
  Attention à la circularité : des étiquettes définies sur les états futurs avantagent la sonde sur les états ;
  `cartographier` définit les régimes futurs sur les **jetons** futurs (prudent), et des étiquettes externes
  (correct / halluciné) sont préférables.
- `ensemble_choc(prompts, t0, n_steps, rang_lo, rang_hi)` : protocole A1 (témoin et choc couplés, jeton de rang imposé).
- `rejouer_moyennes(contextes, suites, couches)` : passe complète, moyenne des états sur la suite (R_h exact de B2).
- `reversibility_analysis(..., detail=True)` : paires « contexte » et « jeton » séparées ; `summarize_patches` donne les
  médianes de chaque type.
- `cartographier(prompts, n, n_steps, layer, t_valeur, rejouer=False)` : point d'entrée de l'atlas ; renvoie un
  dictionnaire sérialisable `{couche, t_valeur, n, n_steps, npc, seed_base, seed_analyse, representation, regimes: {pi,
  pi_ci, n_multi, nul, puissance, verdict}, fiches: [...]}`, une fiche par prompt (`multimodal`, `dip`, `p`,
  `suites_distinctes`, `entropie_moyenne`, et pour les prompts multimodaux `bifurcation` et `valeur_ajoutee`).
  v0.2.1 : `rejouer=True` calcule R_h par une passe rejouée (exactement la représentation de B2) ; `t_valeur` doit
  laisser un jeton avant et deux après.

## 5. Reproduire B2

`python experiments/reproduire_b2.py --model Qwen/Qwen3-1.7B-Base` (puis `HuggingFaceTB/SmolLM3-3B-Base`) regénère B2
à T = 0,6 avec le moteur et compare la décision « multimodal » de chaque prompt à celle publiée. Avec deux GPU,
`--model Qwen/Qwen3-1.7B-Base HuggingFaceTB/SmolLM3-3B-Base` lance un processus par modèle, chacun sur son GPU, en même
temps (sorties `reproduction_b2_<modèle>.json`). L'échantillonneur est
vérifié identique, jeton pour jeton, à `Gen.sample` des notebooks SGZ (`tests/test_v02.py`) ; sur une autre plateforme,
des arrondis fp16 peuvent faire basculer quelques prompts.

## 6. Anciennes expériences

`experiments/multi_prompt.py` et `experiments/replication_akarlar.py` fixent `sampler="multinomial"` et leurs anciens
réglages (T, top-p, gabarit) : relancées, elles refont les mêmes tirages qu'en v0.1.5.

## 7. Tests

`python -m pytest -q tests` : 46 tests du moteur (29 dans `test_trajectory_bifurcation_core.py`, les tests de v0.1.5
avec les deux échantillonneurs ; 17 dans `test_v02.py`), plus les 31 de l'atlas.
Seul le tokenizer de `Qwen/Qwen3-0.6B` est téléchargé.

## 8. Limites connues

- La sortie de la dernière couche après la norme finale (`hidden_states[L]`) n'est pas capturée par les hooks.
- Le lecteur de jetons de `valeur_ajoutee_sur_jetons` est simple (sac haché + 3 derniers jetons) : un écart positif dit
  que le calcul du modèle rend l'issue linéairement lisible au-delà de ce lecteur, pas au-delà de toute lecture du texte.
- Le patching reste à une seule séquence à la fois.
- Modèles chat : avec Gumbel, les jetons spéciaux (dont la fin de tour) sont interdits par défaut, comme pour les
  modèles de base de SGZ ; pour un modèle chat, `forbid_special=False` permet la fin de texte, et `Ensemble.lengths`
  donne la longueur utile de chaque suite (les jetons générés après sont à ignorer).
- `cartographier` ne garde les états qu'au pas `t_valeur` (mémoire) ; `generer_ensemble(..., steps_to_keep=[...])`
  fait de même pour d'autres analyses.
- `multi_prompt.py` fixe aussi les anciens seuils des rapports de recovery (1e-4 et 1e-3) pour refaire ses chiffres.
