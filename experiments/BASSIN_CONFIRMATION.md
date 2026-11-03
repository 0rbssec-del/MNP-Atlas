# Test de confirmation du bassin : témoin même classe, couche fixée, bootstrap des prompts

Écrit le 3 nov. 2026, **avant toute exécution**. Le commit de ce fichier date les critères.

## Pourquoi

La relecture adverse de la note de réplication (3 nov.) a trouvé trois failles dans le test direct du bassin
(`bassin_akarlar.py`, résultats du 2 et du 3 nov.) :

1. le bootstrap tenait les prompts pour fixes ; en les rééchantillonnant, les intervalles de A s'élargissent à environ ±0,4 ;
2. le patch persistant abîme le texte (part « Autre » de 0,14 à 0,26 sur Qwen2.5-7B), sans témoin pour séparer
   « aller vers la source » de « casser la suite » ;
3. sur Qwen2.5-7B, A est positif aux couches tardives (L24 +0,29, non corrigé), seul indice dans le sens de l'article.
   Avec le bootstrap des prompts, cet indice vaut +0,29 [−0,09 ; +0,63] (calcul fait avant ce test).

Ce test vise ces trois points, sur le modèle et la couche de l'indice.

## Montage

- **Modèle** : `Qwen/Qwen2.5-7B` (base), float16 réparti sur deux T4.
- **Prompts** : les **61 prompts** de l'article (`data/akarlar_all_prompts.json`, extraits sans exécution de son dépôt,
  licence MIT), et non plus les seuls 27 qui bifurquent chez Qwen2.5-1.5B. Seuls ceux qui donnent au moins une paire
  discordante dans un réservoir de 60 trajectoires servent ; plus de prompts réduisent l'incertitude due aux prompts.
- **Graine** : 20261105 (échantillon frais ; aucune donnée antérieure réutilisée).
- Comme `bassin_akarlar.py` : T = 0,7, 15 jetons au plus, classifieur d'Akarlar, jusqu'à 6 paires par prompt,
  24 graines par paire et par direction, cible reconstruite, patch persistant, nombres aléatoires communs.
- **Conditions** :
  - base, échange de jeton, auto-patch à L24 ;
  - aux couches 20, 24 et 27, deux patchs :
    - **croisé** : la source est un run de l'autre classe ;
    - **même classe** : la source est un run de la **même classe** que la cible, avec un premier jeton différent. C'est le
      premier run de la réserve qui remplit ces conditions. Ce patch mesure la perturbation due au patch lui-même.

## Critères

**Statistique principale, à la couche fixée L24.** Les deux effets se calculent sur les unités (paire, direction) qui ont
une source même classe :

    E_C→H = logit p_H(croisé) − logit p_H(même classe)    sur les cibles correctes ;
    E_H→C = logit p_C(croisé) − logit p_C(même classe)    sur les cibles hallucinées ;
    A*    = E_C→H − E_H→C.

L'article prédit que corrompre est plus facile que corriger, donc A* > 0.

**IC95** : bootstrap à deux niveaux, d'abord les prompts, puis les paires dans chaque prompt (2000 tirages).

**Verdict** :
- NON TESTABLE si moins de 40 unités par direction ont une source même classe ;
- NON INTERPRÉTABLE si l'auto-patch est identique à la base dans moins de 98 % des suites, ou si l'échange de jeton n'a
  une borne basse > 0 dans aucune direction ;
- **ASYMÉTRIE COMME L'ARTICLE** si la borne basse de A* à L24 > 0 ;
- **ASYMÉTRIE INVERSE** si la borne haute < 0 ;
- **SYMÉTRIQUE** si l'IC de A* tient dans [−0,5 ; +0,5] ;
- **INDÉTERMINÉ** sinon.

**Descriptifs** : A* à L20 et L27 ; A sans témoin (croisé − base) avec bootstrap des prompts, pour voir si l'indice de
L24 se reproduit ; E par direction ; part « Autre » par condition ; version stricte du classifieur.

## Puissance (avant exécution)

Sur les données Qwen2.5-7B du 3 nov., A à L24 (croisé − base) a une demi-largeur de 0,355 avec 25 prompts et le bootstrap
des prompts. Si 35 à 45 prompts donnent des paires, on attend environ 0,26 à 0,30 [hypothèse : variance par prompt
comparable]. A* ajoute le bruit du patch même classe, ce qui donne plutôt 0,30 à 0,40.

Conséquence : le verdict SYMÉTRIQUE n'est atteignable que si A* est proche de 0, et un effet de +0,29 donnerait
probablement INDÉTERMINÉ. Ce test peut confirmer un effet clairement plus grand que +0,4, ou montrer qu'il disparaît
avec le témoin. Il ne peut pas trancher un effet de 0,1 à 0,3.

## Prédictions de l'assistant (écrites avant exécution) [spéculation]

| Verdict | Probabilité |
|---|---|
| ASYMÉTRIE COMME L'ARTICLE | 10 % |
| ASYMÉTRIE INVERSE | 10 % |
| SYMÉTRIQUE | 35 % |
| INDÉTERMINÉ | 35 % |
| NON TESTABLE ou NON INTERPRÉTABLE | 10 % |

Sur la cause de l'indice de L24, je donne 60 % à la perturbation : A* nettement plus petit que A sans témoin.

## Limites

- Le classifieur lexical d'Akarlar n'est toujours pas validé. Une validation (étiquetage manuel d'un échantillon) est
  hors de ce test.
- La source même classe a un premier jeton différent de celui de la cible, comme la source croisée. Elle diffère pourtant
  de la cible par d'autres traits que la classe ; le témoin ne neutralise que la perturbation commune.
- Une seule couche décide ; les autres sont descriptives.

## Commande

    python experiments/bassin_confirmation.py --model Qwen/Qwen2.5-7B --repartir --lot 96 --out bassin_confirmation.json
