# Audit GSM8K : résultat (5 nov. 2026)

Plan : `experiments/AUDIT_GSM8K.md` (cb9cef7, amendements e0e0fa3, afc7cb4). Analyses : `experiments/analyse_gsm8k.py`
→ `analyse.json`. Étiquettes à l'aveugle d'Erick : `etiquettes_erick.txt` (150 réponses : 108 JUSTE, 36 FAUSSE, 6 AUCUNE).

## Verdict selon la lettre du plan : CLASSEMENT DÉPENDANT DU CORRECTEUR, mais c'est un artefact de l'estimateur

Le verdict vient du critère « écart d'au moins 5 points entre extracteur et correction ». Cet écart est produit par
l'estimateur corrigé du plan, pas par les extracteurs : l'estimateur remplace, en strate d'accord, le verdict automatique
par le taux de JUSTE de 35 à 48 réponses tirées au hasard par modèle. Il jette l'information des extracteurs alors
qu'aucune erreur n'y a été trouvée, et son IC95 fait environ ±14 points. Ce défaut de conception a été écrit par
l'assistant avant calcul ; il est déclaré ici, sans changer le verdict pré-enregistré.

## Résultats

**Erreur des extracteurs** (étiquette manuelle, AUCUNE comptée fausse) :

| | Strate D (20, recensement) | Strate A (130 sur 1 480) | Global stratifié (1 500) |
|---|---|---|---|
| strict | 18/20 | 0/130 (borne haute IC95 2,8 %) | 1,2 % (borne haute 4,1 %) |
| flexible | 2/20 | 0/130 (borne haute IC95 2,8 %) | 0,13 % (borne haute 3,2 %) |

- strict : 17 des 18 erreurs sont des réponses justes de Qwen3-1.7B-Base sans « #### » (« The answer is 4. ») ; la 18e
  est « #### 2:30 pm » notée juste parce que la réponse attendue est 2 (étiquetée AUCUNE).
- flexible : 2 erreurs, deux réponses coupées ou ambiguës où le dernier nombre est la bonne réponse (étiquetées AUCUNE).

**Exactitude (500 problèmes) et écarts entre modèles**, IC95 bootstrap apparié :

| | strict | flexible | recensement (secondaire) | plan (pré-enregistré) |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0,628 | 0,628 | 0,626 | 0,539 [0,394 ; 0,685] |
| Qwen3-1.7B-Base | 0,662 | 0,698 | 0,696 | 0,778 [0,640 ; 0,911] |
| SmolLM3-3B-Base | 0,680 | 0,680 | 0,680 | 0,809 [0,681 ; 0,915] |
| Qwen2.5 − Qwen3 | −3,4 [−8,2 ; +1,4] | −7,0 [−11,6 ; −2,4] | −7,0 [−11,6 ; −2,4] | −23,8 [−43,2 ; −4,0] |
| Qwen2.5 − SmolLM3 | −5,2 [−9,6 ; −0,6] | −5,2 [−9,8 ; −0,6] | −5,4 [−10,0 ; −0,8] | −26,9 [−45,6 ; −8,1] |
| Qwen3 − SmolLM3 | −1,8 [−6,2 ; +2,4] | +1,8 [−2,4 ; +6,0] | +1,6 [−2,6 ; +5,8] | −3,1 [−20,5 ; +14,4] |

**Secondaire, non pré-enregistré** : l'estimateur « recensement » garde le verdict automatique en strate A (0 erreur sur
130) et l'étiquette en strate D (entièrement étiquetée). Avec lui, appliqué aux mêmes critères :
- erreur au plus 2 % pour les deux extracteurs (estimation ponctuelle) ;
- l'ordre Qwen3 / SmolLM3 change entre strict (−1,8) et la correction (+1,6), sans que l'IC95 exclue 0 ; l'écart change de
  3,4 points (< 5) ;
- l'écart Qwen2.5 − Qwen3 double avec strict par rapport à la correction (−3,4 contre −7,0), et seul l'écart corrigé exclut 0.

→ **INDÉTERMINÉ** avec cet estimateur : l'ordre bouge, mais sans signification statistique.

## Ce qui est établi

1. Sur ces trois modèles de base, la règle flexible de lm-evaluation-harness s'accorde avec l'étiquetage humain sur
   1 498 des 1 500 réponses estimées ; la règle strict sur 1 482.
2. L'erreur de strict est concentrée sur un modèle (Qwen3-1.7B-Base, 17 réponses justes rejetées, 3,4 points) : elle
   change la conclusion « Qwen2.5 et Qwen3 sont-ils distinguables ? » (IC95 qui contient 0 avec strict, l'exclut corrigé).
3. La strate d'accord ne peut pas prouver une erreur ≤ 2 % : avec 0/130, la borne haute reste à 2,8 %.

## Limites

- Un seul étiqueteur ; 6 AUCUNE, toutes comptées fausses.
- Témoin de relance de Qwen2.5-1.5B (64 %) : surtout un artefact de comparaison (jetons de remplissage de fin), voir `RELANCE.md`.
- Estimateur pré-enregistré inadapté (ci-dessus).

## Travaux voisins

Lyu (2026, arXiv 2610.20639) et Feng, Zhang et Chen (2026, arXiv 2608.25654), voir l'amendement 1. Ce résultat ajoute la
validation humaine à l'aveugle des deux règles GSM8K et leur effet sur l'ordre de modèles de base proches.
