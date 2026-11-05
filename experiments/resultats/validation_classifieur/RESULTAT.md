# Validation du classifieur : résultat (5 nov. 2026)

Plan fixé avant de voir les textes : `experiments/VALIDATION_CLASSIFIEUR.md` (commit f63e55a). Étiquetage à l'aveugle
par Erick (160 suites, `etiquettes_erick.txt`) ; clé révélée après (`cle_cachee.json`).

| Mesure | Valeur | Seuil fixé d'avance |
|---|---|---|
| Accord global | 0,644 | — |
| Kappa de Cohen (3 classes) | **0,315** | FIABLE ≥ 0,60 ; NON FIABLE < 0,40 |
| Précision CORRECT du classifieur | **21/49 = 0,43** [0,30 ; 0,57] | FIABLE ≥ 0,75 ; NON FIABLE < 0,60 |
| Précision HALLUCINATION | 82/85 = 0,96 [0,90 ; 0,99] | ≥ 0,75 |
| Écart différentiel croisé − même classe (H→C) | −0,21 [−0,86 ; +0,44], sur 7 et 4 suites | IC excluant 0 |

**Verdict selon les critères : CLASSIFIEUR NON FIABLE** (kappa < 0,40 et précision CORRECT < 0,60).

Erreur typique : le marqueur « not » suffit à étiqueter CORRECT une suite qui accepte la fausse prémisse
(« Benjamin Franklin, the first President …, did not have a political party »).

Réserves : un seul étiqueteur, rapide ; quelques étiquettes manuelles sont discutables (suite incomplète étiquetée
HALLUCINATION au lieu d'AUTRE). La faible précision de CORRECT est visible indépendamment de l'étiqueteur dans les
exemples. L'écart différentiel n'est pas estimable (trop peu de suites par case).

Conséquence fixée d'avance : A* du test de confirmation (qui repose sur les étiquettes CORRECT pour H→C) est déclaré
non interprétable dans la note ; la critique de l'expérience publiée reste.
