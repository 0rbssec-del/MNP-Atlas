# Réplication d'Akarlar (2026), « Hallucination as Trajectory Commitment »

Article : arXiv 2604.15400. Code : https://github.com/akarlaraytu/trajectory-commitment
(`experiments/e07_trajectory_analysis/trajectory_patching.py`).
Script : `experiments/replication_akarlar.py`. Données brutes :
`experiments/resultats/replication_akarlar.json` et
`experiments/resultats/replication_akarlar_p1.json`.

## Conclusion

Dans notre réplication, le patch de l'article n'a **aucun effet mesurable** :
les taux de « correction » et de « corruption » égalent les taux de base sans
patch. L'asymétrie centrale de l'article (corruption facile, correction
difficile) correspond aux **taux de base** du classifieur, pas à une dynamique
interne. Le profil par couche publié ne dépend pas de la couche (P1b) ; le test
d'égalité exacte en float32 (P1) est en cours.

## Montage

Repris de leur code : Qwen2.5-1.5B (base), prompt brut sans BOS (comme
TransformerLens 2.16.1 pour Qwen2), T = 0.7 sans top-p, 15 tokens au plus,
arrêt sur EOS ou deux retours à la ligne dans les 3 derniers tokens, leurs
8 prompts « les plus équilibrés » (indices 20, 24, 10, 16, 22, 25, 27, 29),
leur classifieur par sous-chaînes, 3 essais par cellule (24 par couche).

Différences : transformers au lieu de TransformerLens ; bfloat16 sur CPU au lieu
de float16 sur MPS pour les phases échantillonnées ; P1 en float32.

Critères fixés avant exécution : commit `5dd1a6e`. Amendements (dtype, P1 à
partir des tokens t0, tirages de complément comme dans leur code) : commit
`bdec38e`, décidés après un test rapide sur un prompt et avant tout résultat
sur P1, P1b ou P2.

## Ce que fait leur code (lecture)

1. **Patch non persistant.** Chaque pas recalcule toute la séquence sans
   cache ; le patch n'est appliqué qu'au passage du pas visé. Au pas suivant,
   son seul effet restant est le token tiré.
2. **Cible jamais reconstruite.** `generate_with_patch` repart du prompt avec
   un nouveau tirage : « halluciné → correct » patche un échantillon neuf,
   pas un run halluciné.
3. **Pas 1 = contexte identique.** Au pas 1, le contexte avant la position
   patchée est le prompt, identique pour tous les runs. Remplacer l'état de la
   dernière position à la couche l rend donc la distribution du pas 1 égale à
   celle de la source, quelle que soit l.

## Résultats

### P1 : la distribution patchée est celle de la source à toutes les couches

En cours (float32, 8 prompts) : résultats à venir dans `replication_akarlar_p1.json`.

### P1b : leur balayage échantillonné ne dépend pas de la couche — CONFIRMÉ

| Direction | Taux par couche (L0, 3, 6, 9, 12, 15, 18, 21, 24, 27), n = 24 | Global | χ² homogénéité |
|---|---|---|---|
| H→C (correct) | 2, 2, 3, 1, 7, 3, 3, 2, 4, 3 | 30/240 = 0.12 [0.09, 0.17] | p = 0.42 |
| C→H (halluciné) | 14, 16, 14, 18, 16, 15, 15, 15, 17, 13 | 153/240 = 0.64 [0.57, 0.70] | p = 0.93 |

L'article rapporte en moyenne 0.16 (H→C) et 0.685 (C→H) ; notre réplication
retrouve ces niveaux.

### Taux de base et asymétrie (descriptif)

| | Sans patch (n = 192) | Leur patch | Écart |
|---|---|---|---|
| Correct | 29/192 = 0.15 [0.11, 0.21] | H→C 0.12 | p = 0.43 |
| Halluciné | 135/192 = 0.70 [0.64, 0.76] | C→H 0.64 | p = 0.15 |

Leur témoin sans patch (10.4 % de corrects sur 15 tokens) concorde avec nos
15 %. Les 36 % de leur tableau 3 portent sur 30 tokens (nous : 37 %).

### P2 : avec un patch persistant, le profil décroît avec la couche — CONFIRMÉ

Cible reconstruite (préfixe du run cible), patch au pas 1 maintenu dans le
KV-cache, continuation de la source forcée ; log-vraisemblance récupérée sur
les pas k ≥ 1 (médiane sur 14 paires par direction dont le premier token
diffère) :

| Couche | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| H→C | 0.98 | 0.87 | 0.80 | 0.59 | 0.41 | 0.34 | 0.23 | 0.09 | 0.00 | 0.00 |
| C→H | 0.97 | 0.78 | 0.71 | 0.62 | 0.34 | 0.28 | 0.09 | 0.06 | 0.00 | 0.00 |

Spearman ρ(couche, médiane) = −1.00 dans les deux directions. Patcher tôt
revient à échanger le token ; patcher tard n'agit que sur le token suivant.
Aucune couche « d'engagement » particulière (L20, L24) n'apparaît.

### Le premier token n'engage presque pas l'issue (descriptif)

- En repartant du premier token d'un run « correct » : 19/96 = 0.20
  [0.13, 0.29] de corrects (base 0.15).
- En repartant du premier token d'un run « halluciné » : 72/96 = 0.75
  [0.65, 0.83] d'hallucinés (base 0.70).
- 10 paires sur 24 (correct, halluciné) partagent le même premier token.

### Classifieur (descriptif)

Seules 2 étiquettes « correct » sur 29 viennent d'une sous-chaîne dans un autre
mot ; le classement strict (mots entiers) change peu les taux. Le problème est
de validité, pas de sous-chaîne : « correct » signifie surtout « contient
"not", "actually"… » et « halluciné » « contient "the", "it", "is"… ».

### Fidélité (descriptif)

Taux de corrects sur 30 tokens, N = 20 par prompt : 0.37 contre 0.36 dans
l'article (Spearman entre prompts +0.10, bruit binomial important à N = 20).

## Limites

- transformers en bf16 sur CPU, et non TransformerLens en fp16 sur MPS ;
  graines différentes.
- 8 prompts, 24 essais par couche, un seul modèle.
- Non répliqués : balayage des pas, fenêtres de patch (attendues triviales vu
  le point 1), sonde de « régime » au pas 0 (section 5.3 de l'article).
- Les étiquettes restent celles de leur classifieur lexical.
