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

## Amendement 1 (5 nov. 2026, avant tout calcul) : antériorité lue et version des extracteurs

**Travaux voisins lus (alphaXiv) :**

- **Lyu (2026), arXiv 2610.20639, « Same Answer, New Box ».**
  - Ce qu'ils font : ils re-notent les sorties de paires de modèles base / RL avec les extracteurs de six dépôts publics,
    dont les deux règles GSM8K de lm-evaluation-harness (commit `b733598`), et avec un extracteur fixe à eux.
  - Ce qu'ils trouvent : sur GSM8K et MATH-500, la règle flexible de lm-evaluation-harness sous-estime le gain de 23,5 %
    du gain lu par leur extracteur fixe. Les extrémités des gains rapportés diffèrent d'une médiane de 15 points selon
    l'extracteur.
  - Ce qu'ils ne font pas : ils préparent un échantillon aveugle de 120 réponses pour un contrôle humain, sans le lancer.
  - Différence avec ce plan : ils comparent un modèle de base à son successeur RL et mesurent une part de gain. Ce plan
    compare des modèles de base entre eux et valide les extracteurs contre des étiquettes humaines à l'aveugle.
- **Feng, Zhang et Chen (2026), arXiv 2608.25654.** Sur des questions ouvertes (NQ, TriviaQA), la correspondance exacte
  rate 42 % des réponses correctes. Des erreurs de notation suffisent à inverser une décision en aval (choix d'une règle
  de calibration), mesuré contre des étiquettes humaines. Pas de GSM8K.

**Conséquences pour ce plan, sans changement des critères ni des prédictions :**

1. **Les extracteurs** sont ceux de lm-evaluation-harness au commit `b733598`, comme dans Lyu (2026) :
   - **strict** : le premier « #### nombre » ;
   - **flexible** : la dernière suite d'au moins deux caractères parmi chiffres, « $ », points et virgules, sinon de
     chiffres.

   Une réponse sans correspondance rend « [invalid] », compté faux. Copies verbatim et tests contre la source avant
   exécution.
2. **La note finale citera ces deux travaux.** La contribution annoncée se limite à ce qu'ils ne font pas :
   - la validation humaine à l'aveugle des deux extracteurs de GSM8K ;
   - l'effet sur le classement de modèles de base proches.

## Amendement 2 (5 nov. 2026, avant tout calcul) : fidélité à lm-evaluation-harness

Pour que la notation automatique soit exactement celle de l'outil audité, trois points du plan s'alignent sur le fichier
`gsm8k.yaml` et sur `exact_match_hf_evaluate` au commit `b733598` :

1. **Comparaison** : correspondance exacte de chaînes après les `regexes_to_ignore` de l'outil (« , », « $ »,
   « (?s).*#### », « \.$ ») et passage en minuscules, et non comparaison numérique. La comparaison numérique reste un
   descriptif.
2. **Arrêt de la génération** : les marqueurs de l'outil (« Question: », « </s> », « <|im_end|> »), sans le double saut
   de ligne prévu au départ ; 256 jetons au plus.
3. **Exemples** : les 5 premiers problèmes du jeu d'entraînement, au format « Question: … / Answer: … » de l'outil (qui,
   lui, tire ses exemples au hasard).

**Tests avant exécution.** Le script `experiments/audit_gsm8k.py` vérifie 20 réponses synthétiques contre la logique
copiée de la source. Au premier essai, il a signalé deux valeurs attendues fausses, écrites par l'assistant : la règle
flexible prend la dernière correspondance, « 2 » dans « value 10 and 2 » et « 3 » dans « costs 2, 3 ». Le code était
conforme, et les valeurs attendues ont été corrigées.

## Amendement 3 (5 nov. 2026, après l'étape 1, avant ce test) : témoin de relance échoué pour Qwen2.5-1.5B

**Constat.** Relance identique à 64 % (32/50) pour Qwen2.5-1.5B, contre 98 % et 100 % pour les deux autres. Le témoin
relançait les 50 premiers problèmes en lots de 32 : le premier lot (1–32) avait la même composition que dans la passe
principale, le second (33–50) non (18 problèmes au lieu de 32, donc un remplissage à gauche différent).

**Hypothèse testée.** Les écarts viennent de la composition des lots (forme du calcul en float16), et non d'un
non-déterminisme à lot identique. Ce test ne change ni l'échantillon d'étiquetage ni les analyses de l'étape 2, qui portent
sur les sorties figées de l'étape 1.

**Montage** (`experiments/relance_gsm8k.py`, Qwen2.5-1.5B, mêmes prompts, même décodage) :
- R1a, R1b : problèmes 1–64 en lots de 32, deux fois (mêmes lots que la passe principale) ;
- R2 : problèmes 1–50 en lots de 32 (reproduction du témoin) ;
- R3 : problèmes 1–200 en lots de 1 (aucun remplissage) ; les 20 premiers relancés une fois (R3b) ;
- secondaire, R4 : R1a et R2 refaits en float32.

**Critères.**
- **COMPOSITION DU LOT** si R1a = R1b sur 64/64, si R2 = R1a sur les 32 problèmes du lot commun, et si au moins un des
  18 problèmes du lot modifié diffère.
- **NON-DÉTERMINISME À LOT FIXE** si R1a ≠ R1b sur au moins un problème.
- **INDÉTERMINÉ** sinon.
- **Effet sur le score** (descriptif, avec critère) : différence d'exactitude (strict et flexible) entre R3 et la passe
  principale sur les 200 problèmes, IC95 par bootstrap apparié. **EFFET DÉTECTÉ** si l'IC95 exclut 0, sinon « effet
  non détecté à n = 200 », avec la largeur de l'IC.
- Descriptif : R1a comparé à la passe principale (même matériel, autre session) ; part des problèmes de R3 dont la
  réponse change ; R4 dit si l'effet de composition persiste en float32.

**Prédiction de l'assistant** [spéculation] : COMPOSITION DU LOT 75 %, NON-DÉTERMINISME À LOT FIXE 15 %,
INDÉTERMINÉ 10 % ; effet sur le score non détecté à n = 200 (70 %).
