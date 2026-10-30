# Code atlas et moteur v0.2

## Séparation

| Partie | Fichiers | Règle |
|---|---|---|
| Moteur | `trajectory_bifurcation_core.py`, `MOTEUR_V02.md`, `tests/test_v02.py`, `tests/test_trajectory_bifurcation_core.py`, `experiments/reproduire_b2.py` | Livré tel quel (commit `d900d27`) ; le code atlas ne le modifie pas. |
| Couche atlas | `atlas/compat.py` | Seul point de contact avec le moteur : protocoles figés, accès aux méthodes internes, garde de version. |
| Expériences atlas | `experiments/multi_prompt.py`, `experiments/replication_akarlar.py` | Construisent leur moteur avec `construire_moteur(protocole, ...)` et n'appellent aucune méthode interne directement. |

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

## Note sur MOTEUR_V02.md

`MOTEUR_V02.md` attribue le témoin négatif à recouvrement 0 de `resultats_multi_prompt.json` à « la position d'un seuil
cumulé dans `torch.multinomial` » (cause probable). La cause vérifiée est autre : les deux continuations de ce témoin
étaient identiques (`first_divergence_step` = None) mais vides (fin de texte immédiate), et le Jaccard de deux ensembles
vides valait 0. Corrigé dans le commit `bdec38e`, correction reprise par le moteur v0.2 (« deux continuations vides sont
identiques »). L'échantillonneur Gumbel garde ses autres avantages (nombres aléatoires communs entre branches).
