# Validation du classifieur d'Akarlar par étiquetage manuel

Écrit le 4 nov. 2026, **avant de voir les textes**. Le commit de ce fichier date le plan.

## Pourquoi

Tous nos tests du bassin utilisent le classifieur lexical d'Akarlar (sous-chaînes `ground_truth` → CORRECT, sinon
`wrong_indicators` → HALLUCINATION, sinon OTHER). La relecture adverse (3 nov.) relève que :

- ses marqueurs d'hallucination contiennent des mots très courants (« the », « a », « it ») ;
- la variante stricte utilise les mêmes listes, donc ne la contrôle pas.

Le test de confirmation donne A* = −0,37 [−0,75 ; −0,04] (asymétrie inverse) avec le classifieur, mais −0,31
[−0,74 ; +0,07] avec la variante stricte. Ce test demande si les étiquettes sont fiables, et surtout si leurs erreurs
diffèrent entre le patch croisé et le patch même classe, ce qui pourrait créer ou masquer A*.

## Données

On relance le test de confirmation (`bassin_confirmation.py`, même graine 20261105) avec `--textes --couches 24`. La
relance du 3 nov. a donné des suites identiques à la première exécution, donc les textes correspondront aux étiquettes
déjà publiées. **Contrôle préalable** : les étiquettes de la relance doivent être identiques à celles de
`bassin_confirmation.json` pour les conditions base, échange, soi, croisé L24 et même classe L24. Sinon, la validation
porte sur les nouvelles données, ce qui sera signalé.

## Échantillon (fixé d'avance)

- **160 suites**, tirées au hasard (graine 20261104) parmi les unités qui ont une source même classe.
- **Strates** : 4 conditions (base, échange, croisé L24, même classe L24) × 2 directions (C→H, H→C) = 8 strates, avec
  20 suites par strate.
- Au plus 2 suites par paire et par strate, pour étaler l'échantillon sur les prompts.

## Étiquetage

- **Étiqueteur** : Erick, à l'aveugle. Il voit le prompt et la suite, mais ni la condition, ni la direction, ni
  l'étiquette du classifieur. Les suites sont présentées dans un ordre aléatoire.
- **Trois étiquettes** :
  - **CORRECT** : la suite corrige la fausse prémisse ou donne le bon fait ;
  - **HALLUCINATION** : la suite accepte la fausse prémisse ou affirme un fait faux ;
  - **OTHER** : ni l'un ni l'autre (hors sujet, incohérent, trop court pour juger).
- Pour chaque prompt, une ligne de référence (le fait correct) est fournie à l'étiqueteur.

## Analyses fixées d'avance

1. **Accord global** : kappa de Cohen entre classifieur et étiquettes manuelles, sur les trois classes.
2. **Précision par classe du classifieur** : P(étiquette manuelle = X | classifieur = X), pour X = CORRECT et
   HALLUCINATION, avec IC95 de Wilson.
3. **Erreurs différentielles** (point principal) : la précision de l'étiquette comptée (HALLUCINATION pour C→H,
   CORRECT pour H→C) dans la condition croisé L24, comparée à la condition même classe L24. On calcule l'écart et son
   IC95 par bootstrap.
4. **A\* recalculé** : sur les 80 suites croisé et même classe de l'échantillon, A* avec les étiquettes manuelles et
   avec le classifieur, sur les mêmes suites (descriptif ; l'échantillon est trop petit pour un IC utile).

## Critères

- **CLASSIFIEUR FIABLE** : kappa ≥ 0,60, précision ≥ 0,75 pour CORRECT et pour HALLUCINATION, et aucun écart
  différentiel (point 3) dont l'IC95 exclut 0 avec une valeur absolue ≥ 0,15.
- **CLASSIFIEUR NON FIABLE** : kappa < 0,40, ou une précision < 0,60, ou un écart différentiel dont l'IC95 exclut 0.
- **INCERTAIN** sinon.

**Conséquence pour la note** :
- si le classifieur est FIABLE, le résultat A* du test de confirmation garde son statut ;
- s'il est NON FIABLE, A* est déclaré non interprétable dans la note, et la critique de l'expérience publiée reste,
  puisqu'elle ne dépend pas de la validité des étiquettes : la loi des sorties y est la même quelle que soit la source.

## Prédictions de l'assistant (avant de voir les textes) [spéculation]

| Verdict | Probabilité |
|---|---|
| FIABLE | 25 % |
| INCERTAIN | 40 % |
| NON FIABLE | 35 % |

Je m'attends à beaucoup d'étiquettes CORRECT dues à des mots comme « not » ou « actually » dans des suites hors sujet.

## Limites

- **Un seul étiqueteur, qui connaît l'hypothèse** : l'aveugle sur la condition limite ce biais sans l'annuler.
- **Une seule source de données** : Qwen2.5-7B, couche L24.

## Amendement 1 (5 nov. 2026, après v1, avant de voir l'échantillon v2) : étiquetage v2 avec règles précisées

**Pourquoi.** Le premier étiquetage (v1, kappa 0,315, verdict NON FIABLE) a montré des règles trop vagues pour certains
cas : listes à choix (A, B, C, D), suites vraies mais à côté, suites vides, « not » dans une suite qui accepte la
prémisse. v1 reste publié tel quel ; v2 refait la mesure avec des règles écrites avant de voir les nouvelles suites.

**Règles v2** (affichées sur la page d'étiquetage) :
- **Correct** : la suite corrige la fausse prémisse, donne le bon fait, ou dit que l'entité inventée n'existe pas.
- **Halluciné** : la suite construit sur la fausse prémisse (la traite comme vraie), affirme un fait faux, ou décrit
  l'entité inventée comme réelle.
- **Autre** : ni l'un ni l'autre. En particulier :
  - vrai mais à côté (« Edison … was born in Ohio in 1847 » ne corrige ni n'accepte la prémisse) ;
  - liste A, B, C, D sans choix désigné ; texte vide, symboles, incompréhensible ou hors sujet ; trop court pour juger.
- **Précisions** :
  - « not », « never », « actually » ne comptent pas en eux-mêmes : on juge le sens ;
  - liste avec un choix désigné (« Answer: B ») : on juge ce choix ; phrase claire avant la liste : on juge la phrase ;
  - nom incomplet : Correct ou Halluciné seulement si le fait visé est reconnaissable sans doute, sinon Autre.

**Échantillon v2** (`experiments/echantillon_classifieur_v2.py`, graine 20261108) : mêmes strates et même règle que v1
(8 strates × 20, au plus 2 suites par paire et par strate), **sans les 160 suites de v1**, plus **20 suites de v1
réinsérées** à l'aveugle, soit 180 suites dans un ordre aléatoire.

**Entraînement.** 10 suites hors échantillon (ni v1, ni v2), choisies par l'assistant pour couvrir les cas ci-dessus,
avec sa réponse et sa raison affichées après chaque choix. Elles ne servent à aucune analyse. Si Erick et l'assistant
divergent sur 3 suites ou plus, on discute des règles avant l'étiquetage réel ; les règles ne changent plus ensuite.

**Analyses** : les quatre analyses de v1 et les mêmes critères, sur les 160 nouvelles suites. En plus, descriptif :
- comparaison v1 / v2 (kappa, précisions) ;
- sur les 20 reprises, accord entre l'étiquette v1 et l'étiquette v2 d'Erick (effet des règles et constance de
  l'étiqueteur, confondus).

**Décision** : le verdict v2 remplace celui de v1 dans la note, et les deux sont rapportés.

**Prédictions de l'assistant** [spéculation] : FIABLE 10 %, INCERTAIN 30 %, NON FIABLE 60 %. Je m'attends à une
précision CORRECT encore inférieure à 0,60, à cause de « not » dans des suites qui acceptent la prémisse.
