# Code atlas et moteur v0.2

## Atlas v0.4 : cartes de l'espace caché

L'atlas v0.3 indexait les mesures v0.1 (t* JS, `has_bifurcation`, recovery au premier pas). Les résultats SGZ et la
réplication d'Akarlar montrent qu'elles suivent la divergence des jetons (C1 : 16 runs sur 16) et l'identité du dernier
jeton, pas une structure interne. La v0.4 fait des **cartes de l'espace caché** le type d'entrée principal et garde les
entrées v0.1 comme historiques (API et valeurs inchangées).

| | |
|---|---|
| Construire | `atlas.cartographier("sgz", prompts, moteur_kwargs=..., n=192, n_steps=32, rejouer=True)` (ou un moteur déjà construit) : `cartographier` du moteur (core ≥ 0.2.1), provenance complète |
| Une carte | régimes de B2 sur R_h (test par moitiés, substitut gaussien, mélanges, BH, verdict) ; une fiche par prompt : `multimodal`, `dip`, `p`, `suites_distinctes`, `entropie_moyenne` ; pour un prompt multimodal, `bifurcation` (préfixe) et `valeur_ajoutee` (état au pas `t_valeur` contre un lecteur de jetons) |
| Interroger | `fiches(i, multimodal=, verdict_valeur=)`, `signature(i, prompt)` (NaN si absent ; non testé ≠ « aucune »), `matrice_signatures(i)`, `resume_carte(i)` |
| Comparer | `accord_cartes(i, j)` : prompts multimodaux partagés, attendu par hasard, p hypergéométrique, Jaccard (B2 après coup : 5 partagés, p = 0,009) ; comparabilité sans le modèle ni la couche |
| Stocker | `atlas_cartes.json` (JSON seul) à côté de `atlas_meta.json` et `atlas_data.pkl` ; `exporter_carte` / `importer_carte` pour des cartes faites dans des processus séparés |
| Kaggle | `experiments/construire_atlas.py --model Qwen/Qwen3-1.7B-Base HuggingFaceTB/SmolLM3-3B-Base` : un modèle par GPU, carte des 64 prompts de B2 au protocole de B2, contrôle 64/64 contre B2, atlas fusionné, accord entre modèles. Remplace `reproduire_b2.py` |

**Comparabilité à trois états** (`compare_entries`, `accord_cartes`) : `True` si tous les réglages coïncident,
`False` si l'un diffère (`differences`), `None` si l'un manque (`inconnus`). Réglages comparés : modèle,
échantillonneur, T, top-p, gabarit, jetons spéciaux, **précision de calcul**, seuils de recovery, top-k et
**arguments d'analyse** (réflexion on/off) pour les entrées historiques ; q, couche, `t_valeur`, n, pas, ACP et
représentation pour les cartes. Corrige deux défauts de v0.3 : deux entrées réflexion ON et OFF, ou à seuils de recovery
différents, étaient « comparables » ; la précision de calcul n'était pas enregistrée. Une provenance v0.3 donne donc
`None` (précision inconnue) là où v0.3 répondait `True`.

**Provenance** (`reglages_moteur`) : ajoute précision de calcul, matériel, jetons spéciaux, q, top-k, couches, taille
de lot, nombre de couches, versions de torch et transformers.

**Limites.** La carte ne garde que des résumés par prompt (pas les états) ; la valeur ajoutée a besoin d'au moins 5
trajectoires par régime (sinon NON TESTABLE) ; les étiquettes de régime futur sont définies sur les jetons futurs
(prudent, voir `MOTEUR_V02.md`) ; `accord_cartes` suppose les prompts échangeables (même p que l'analyse après coup de B2).

## Séparation

| Partie | Fichiers | Règle |
|---|---|---|
| Moteur | `trajectory_bifurcation_core.py`, `MOTEUR_V02.md`, `tests/test_v02.py`, `tests/test_trajectory_bifurcation_core.py`, `experiments/reproduire_b2.py` | Livré par la session moteur (v0.2.0, commit `d900d27` ; v0.2.1 avec l'atlas v0.4 : `cartographier(rejouer=…)` et réglages renvoyés avec la carte) ; le code atlas ne le modifie pas. |
| Couche atlas | `atlas/compat.py` | Seul point de contact avec le moteur : protocoles figés, accès aux méthodes internes, garde de version. |
| Expériences atlas | `experiments/multi_prompt.py`, `experiments/replication_akarlar.py` | Construisent leur moteur avec `construire_moteur(protocole, ...)` et n'appellent aucune méthode interne directement. |
| Atlas des résultats | `atlas/bifurcation_atlas.py` (raccourci : `bifurcation_atlas.py`) | v0.4 : cartes de l'espace caché, et `BifurcationResult` historiques ; types, version et réglages du moteur lus via `atlas/compat.py`. |
| Expérience atlas v0.4 | `experiments/construire_atlas.py`, `tests/test_atlas_v04.py` | Première carte (64 prompts de B2, deux modèles) ; 9 tests. |

Si le moteur change d'API, seul `atlas/compat.py` doit suivre ; `verifier_moteur()` refuse un moteur sans l'API v0.2
avec la liste de ce qui manque.

## Protocoles

| Protocole | Échantillonneur | T | top-p | Prompt | Seuils de recovery | Sert à |
|---|---|---|---|---|---|---|
| `v015` | multinomial | 0,8 | 0,95 | gabarit de chat | 1e-4 nat, 1e-3 nat | refaire `resultats_multi_prompt.json` |
| `akarlar` | multinomial | 0,7 | 1,0 | brut | (non utilisés) | réplication d'Akarlar, réglages de leur code |
| `sgz` | Gumbel, graine par ligne et par pas | 0,6 | 1,0 | brut, jetons spéciaux interdits | défauts v0.2 (0,05 nat, 0,5 nat) | analyses atlas au protocole du moteur v0.2 |

Un argument contraire au protocole (par exemple `temperature=0.6` avec `v015`) est refusé : changer de réglage, c'est
changer de protocole.

```python
from atlas.compat import construire_moteur
eng = construire_moteur("sgz", max_new_tokens=48, layers_to_track=[0, 7, 14, 21])   # Qwen/Qwen3-0.6B-Base
```

`experiments/multi_prompt.py --protocole sgz` applique les critères C1 à C4 au protocole v0.2 : prompt brut (pas de mode
réflexion), longueur fixe, et paires informatives scindées en « contexte » et « jeton » (`detail=True`).

## Atlas v0.3 (`bifurcation_atlas.py`)

Adapté de l'atlas v0.2 (écrit pour le core v0.1.4). API et valeurs par défaut inchangées ; ajouts :

- **Provenance par entrée** : `build_from_prompts("sgz", prompts, moteur_kwargs=...)` (ou un moteur déjà construit)
  enregistre protocole, modèle, échantillonneur, T, top-p, gabarit, graine, longueur et version du moteur.
  `summary()` signale les protocoles mélangés ; `compare_entries()` dit si deux entrées sont comparables.
- **Mesures de v0.2** dans `EntrySummary` : `corr_js_divergence_jetons` et `bifurcation_suit_jetons` (≥ 0,9 : la
  « bifurcation » JS suit la divergence des jetons), `t_patch`, nombres de paires « contexte » / « jeton » et médianes de
  `recovery_multistep` et `recovery_loglik` par type.
- **Classements et filtres** : `mesure` (`recovery`, `recovery_multistep`, `recovery_loglik`), `detail`
  (`informative`, `informative_context`, `informative_token`), `agregat` (`mean` par défaut, `median`).
- **Navigation** : valeur absente = NaN (et non plus −1,0, qui est un recovery possible).
- **Compatibilité ascendante** : relit les atlas v0.2 (sans provenance) ; pour une paire picklée avant la v0.2, le type
  détaillé vient de `last_token_same` et non de la valeur par défaut de la classe ; la corrélation C1 est recalculée
  à partir de `js_over_time` et `frac_pairs_same_prefix`.

À lire avec prudence : `has_bifurcation` (t* détecté) et `mean_recovery_informative` (moyenne du recovery au premier
pas, souvent dominé par l'identité du dernier jeton) gardent leur sens de v0.2 pour ne pas changer les chiffres
existants ; les médianes multi-pas par type de paire sont plus fiables. `build_from_prompts` passe par
`analyze_prompt` du moteur, qui ne sépare pas les paires par quota « contexte » / « jeton » : les paires « contexte »
y sont rares.

Le pickle exécute du code au chargement : ne charger que des atlas de confiance.

## Vérification de la compatibilité

- `tests/test_atlas_compat.py` (12 tests, petit Qwen3 aléatoire) : réglages de chaque protocole, refus des réglages
  contraires et d'un moteur sans API v0.2, tirages des protocoles `v015` et `akarlar` identiques à une
  réimplémentation indépendante de l'échantillonneur v0.1.5, gabarit selon le protocole, accès internes, lecture des
  JSON antérieurs à la v0.2, `multi_prompt.run_one` sous `v015` et `sgz`.
- Relance réelle, moteur v0.2 et couche atlas, comparée aux JSON publiés :

  - `multi_prompt.py` (protocole `v015`), premier run publié (« Raconte une histoire courte. », réflexion ON) :
    t* = 7, t_patch = 8, corrélation C1, 24 paires × 10 champs et 8 témoins nuls **identiques**, flottants compris.
  - `replication_akarlar.py` (protocole `akarlar`, bf16), prompt 20 : 20 textes de découverte et 26 runs du pool
    (texte, étiquettes, premier jeton, tirages de complément) **identiques**.
  - `multi_prompt.py --protocole sgz` sur Qwen/Qwen3-0.6B-Base, un prompt (essai de bout en bout, pas un résultat) :
    Gumbel, prompt brut, longueur fixe (48 jetons), 12 paires « contexte » et 16 « jeton », témoin négatif à
    recouvrement 1,0 ; 2 min 30 sur 4 cœurs.
  - Atlas v0.3, `build_from_prompts("sgz", ...)` sur Qwen/Qwen3-0.6B-Base, 2 prompts × 8 trajectoires (essai) :
    construction, sauvegarde, rechargement, résumé, provenance et comparabilité fonctionnent ; 1 min 45.
- `tests/test_bifurcation_atlas.py` : 10 tests (résumés, entrées v0.1.x, provenance, sérialisation, construction en
  lot au protocole `sgz`).

## Note sur MOTEUR_V02.md

`MOTEUR_V02.md` attribue le témoin négatif à recouvrement 0 de `resultats_multi_prompt.json` à « la position d'un seuil
cumulé dans `torch.multinomial` » (cause probable). La cause vérifiée est autre : les deux continuations de ce témoin
étaient identiques (`first_divergence_step` = None) mais vides (fin de texte immédiate), et le Jaccard de deux ensembles
vides valait 0. Corrigé dans le commit `bdec38e`, correction reprise par le moteur v0.2 (« deux continuations vides sont
identiques »). L'échantillonneur Gumbel garde ses autres avantages (nombres aléatoires communs entre branches).
