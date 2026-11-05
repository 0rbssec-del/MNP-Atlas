# Audit du correcteur de GSM8K : l'erreur d'extraction change-t-elle le classement des modèles ?

Écrit le 5 nov. 2026, **avant tout calcul**. Le commit de ce fichier date le plan. Premier audit du groupe « benchmarks
connus » (GSM8K, IFEval, MATH-500), dans le prolongement du registre de reproductibilité et d'erreurs.

## Question

GSM8K est noté en extrayant un nombre de la réponse du modèle et en le comparant à la réponse attendue. L'outil
d'évaluation le plus utilisé (lm-evaluation-harness) propose deux extracteurs, « strict » et « flexible ». Combien de
réponses chacun juge-t-il mal, et ces erreurs changent-elles le classement entre modèles ?

## Antériorité connue (non revérifiée ici)

- Les deux extracteurs existent précisément parce qu'ils donnent des scores différents. Ce qui n'est pas connu de nous :
  une mesure de leur erreur contre un étiquetage manuel à l'aveugle, et son effet sur un classement.
- À vérifier avant exécution : un travail publié qui aurait déjà mesuré l'erreur d'extraction de GSM8K contre des
  étiquettes humaines. S'il en existe un, ce plan devient une réplication, et sera présenté comme telle.

## Montage

- **Modèles (base, ouverts)** : `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen3-1.7B-Base`, `HuggingFaceTB/SmolLM3-3B-Base`. Ils sont de
  taille proche, et leur classement est donc probablement serré, donc sensible au correcteur.
- **Données** : 500 problèmes du jeu de test de GSM8K (1 319), tirés au hasard avec la graine 20261105, la même liste pour
  les trois modèles.
- **Prompt** : 5 exemples fixes tirés du jeu d'entraînement (les mêmes pour tous), format « Question: … / Answer: … ####
  nombre », comme dans le format standard de GSM8K.
- **Génération** : décodage glouton (déterministe), 256 jetons au plus, arrêt sur « Question: » ou deux sauts de ligne.
  float16 sur Kaggle.
- **Extracteurs**, repris de lm-evaluation-harness (expressions à vérifier contre leur fichier de configuration avant
  exécution) :
  - **strict** : le nombre qui suit « #### » ;
  - **flexible** : le dernier nombre de la réponse.

  Les deux sont normalisés de la même façon : virgules et « $ » retirés, nombres comparés numériquement.

## Étiquetage à l'aveugle

- **Échantillon : 150 réponses**, tirées selon deux strates fixées d'avance (graine 20261106) :
  - **D**, désaccord : les réponses où les deux extracteurs rendent des verdicts différents (juste / faux). On en prend
    jusqu'à 90, réparties à parts égales entre modèles autant que possible ;
  - **A**, accord : on complète jusqu'à 150 par un tirage au hasard parmi les réponses où les deux extracteurs sont
    d'accord.
- **L'étiqueteur (Erick) voit** la question, la réponse du modèle et la réponse attendue. **Il ne voit ni le modèle ni
  le verdict des extracteurs.**
- **Il décide** : la réponse finale du modèle est-elle juste (JUSTE), fausse (FAUSSE) ou introuvable (AUCUNE) ?

## Analyses fixées d'avance

1. **Erreur de chaque extracteur** dans chaque strate : part des verdicts contraires à l'étiquette manuelle. Une réponse
   AUCUNE compte comme fausse.
2. **Exactitude corrigée de chaque modèle** : estimateur stratifié. Pour chaque strate, on prend le taux de JUSTE parmi
   les réponses étiquetées, pondéré par la taille de la strate pour ce modèle. IC95 par bootstrap sur les problèmes.
3. **Classement** : différence d'exactitude entre chaque paire de modèles, avec l'extracteur strict, avec le flexible, et
   corrigée par l'étiquetage. IC95 par bootstrap apparié sur les problèmes.

## Critères

- **CORRECTEUR FIABLE** si les deux extracteurs se trompent sur au plus 2 % des réponses (estimation stratifiée sur les
  1 500 réponses), et si aucune paire de modèles ne change d'ordre entre extracteur et correction.
- **CLASSEMENT DÉPENDANT DU CORRECTEUR** si, pour au moins une paire de modèles, l'ordre selon un extracteur est inversé
  par la correction, et si l'IC95 de la différence corrigée exclut 0. Même verdict si la différence entre deux modèles
  change d'au moins 5 points de pourcentage entre extracteur et correction.
- **ERREUR SANS EFFET SUR LE CLASSEMENT** si un extracteur se trompe sur plus de 2 % des réponses sans que les ordres
  changent.
- **INDÉTERMINÉ** sinon.

**Témoins** :
- le décodage glouton relancé deux fois doit donner des réponses identiques (contrôle sur 50 problèmes) ;
- les extracteurs sont testés sur 20 réponses synthétiques à verdict connu.

## Prédictions de l'assistant (écrites avant exécution) [spéculation]

| Verdict | Probabilité |
|---|---|
| CORRECTEUR FIABLE | 15 % |
| ERREUR SANS EFFET SUR LE CLASSEMENT | 45 % |
| CLASSEMENT DÉPENDANT DU CORRECTEUR | 30 % |
| INDÉTERMINÉ | 10 % |

Je m'attends à ce que l'extracteur strict sous-estime surtout les modèles de base, qui oublient souvent « #### ».

## Limites

- **Un seul étiqueteur** (déjà constaté rapide lors de la validation du classifieur d'Akarlar) : la consigne demande de
  marquer AUCUNE quand c'est douteux.
- **Le prompt et le décodage** sont un choix parmi d'autres ; les classements publiés utilisent parfois d'autres réglages.
- **Trois petits modèles de base** : le résultat ne s'étend pas d'office aux grands modèles ni aux modèles à consigne.
