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
