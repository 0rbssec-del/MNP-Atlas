# Test du bassin d'Akarlar sur deux modèles plus gros (Qwen2.5-7B, OLMo-2-7B)

Écrit le 3 nov. 2026, **avant toute exécution**. Le commit de ce fichier date les critères.

## Pourquoi

Sur Qwen2.5-1.5B (le modèle de l'article), le test direct du bassin est symétrique : A = −0,10 [−0,28 ; +0,08] sur les
27 prompts qui bifurquent (`BASSIN_27.md`, résultats `resultats/kaggle_2026-11-03/`). La limite principale de la note de
réplication est : un seul modèle. Ce test demande si le résultat tient sur des modèles plus gros, d'une même famille
(Qwen2.5-7B, même famille que l'article) et d'une autre famille (OLMo-2-7B, données et entraînement ouverts).

## Modèles

- `Qwen/Qwen2.5-7B` (base), 28 couches ;
- `allenai/OLMo-2-1124-7B` (base), 32 couches.

En fp16 (T4 sans bf16), chacun fait environ 15 Go et ne tient pas sur une seule T4 : il est réparti sur les deux GPU
(`device_map="auto"`), et les deux modèles tournent l'un après l'autre.

## Montage et critères

Identiques à `bassin_akarlar.py` et `BASSIN_27.md`, sans changement de règle :

- les 27 prompts qui bifurquent chez Akarlar (`data/akarlar_bifurcating_prompts.json`), son classifieur, T = 0,7,
  15 jetons au plus, réservoir de 60 trajectoires, jusqu'à 6 paires par prompt, 24 graines par paire et par direction ;
- patch persistant, échange de jeton, auto-patch, nombres aléatoires communs, Δ en log-odds, A = Δ_C→H moyen − Δ_H→C
  moyen, IC95 par bootstrap des paires stratifié par prompt ;
- mêmes témoins et mêmes verdicts (ASYMÉTRIE COMME L'ARTICLE, ASYMÉTRIE INVERSE, SYMÉTRIQUE, INDÉTERMINÉ).

Trois adaptations, fixées ici :

1. **Couches à la même profondeur relative** (`--couches-relatives`) : arrondi de l × (N − 1) / 27 pour
   l ∈ {0, 6, 12, 18, 20, 24, 27}. Qwen2.5-7B (28 couches) : mêmes couches ; OLMo-2-7B (32 couches) : 0, 7, 14, 21, 23,
   28, 31, auto-patch à L23.
2. **Seuil de paires** (`--min-paires 40`) : un modèle plus gros connaît mieux les faits, et certains prompts ne
   bifurqueront plus. En dessous de 40 paires discordantes, le verdict est NON TESTABLE pour ce modèle.
3. **Lots de 96** (`--lot 96`) pour la vitesse ; la taille des lots ne change pas les nombres aléatoires (graine par
   ligne et par pas).

Graine : 20261104 pour les deux modèles (échantillon frais).

## Puissance

Si un modèle fournit autant de paires que Qwen2.5-1.5B (162), la demi-largeur attendue de l'IC de A est d'environ 0,18
(observée sur 27 prompts). Avec 40 à 80 paires, elle serait plutôt de 0,25 à 0,35 [hypothèse : variance par paire
comparable], ce qui suffit encore à exclure une asymétrie de la taille annoncée par l'article.

## Prédictions de l'assistant (écrites avant exécution) [spéculation]

Par modèle : ASYMÉTRIE COMME L'ARTICLE 10 %, ASYMÉTRIE INVERSE 15 %, SYMÉTRIQUE 45 %, INDÉTERMINÉ 15 %,
NON TESTABLE 15 %.

## Limites

- Les 27 prompts ont été choisis parce qu'ils bifurquent chez Qwen2.5-1.5B ; chez un autre modèle, les prompts qui
  bifurquent encore forment un sous-ensemble sélectionné.
- Le classifieur lexical d'Akarlar n'est pas validé, et peut se comporter autrement sur les sorties d'un autre modèle.
- La répartition sur deux GPU n'a pas pu être testée localement (CPU seul) ; le reste du montage l'a été.

## Commandes

    python experiments/bassin_akarlar.py --model Qwen/Qwen2.5-7B --prompts experiments/data/akarlar_bifurcating_prompts.json \
        --seed 20261104 --repartir --couches-relatives --lot 96 --min-paires 40 --out bassin_qwen7b.json
    python experiments/bassin_akarlar.py --model allenai/OLMo-2-1124-7B --prompts experiments/data/akarlar_bifurcating_prompts.json \
        --seed 20261104 --repartir --couches-relatives --lot 96 --min-paires 40 --out bassin_olmo7b.json
