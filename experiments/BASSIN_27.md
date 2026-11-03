# Test du bassin d'Akarlar sur les 27 prompts qui bifurquent — réplication de confirmation

Écrit le 3 nov. 2026, **avant toute exécution**. Le commit de ce fichier date les critères.

## Pourquoi

Premier test du bassin (2 nov., 8 prompts, 48 paires, graine 20261101, résultats dans
`resultats/kaggle_2026-11-02/bassin_akarlar.json`) : A = −0,25 [−0,51 ; +0,02], INDÉTERMINÉ. La largeur de l'intervalle
vient surtout du nombre de prompts : le bootstrap est stratifié par prompt et la variance passe par eux.

Ce test-ci est une **réplication de confirmation** : un échantillon frais (nouvelle graine, nouveaux réservoirs, nouvelles
paires), sur les **27 prompts qui bifurquent** chez Akarlar (20 tirages à T = 0,7, au moins 2 corrects et 2 hallucinés,
fichier `bifurcation_Qwen_Qwen2.5-1.5B.json` de son dépôt). Les données du 2 nov. ne sont **pas** ajoutées : elles ont
servi à choisir la taille de l'échantillon.

## Données

`data/akarlar_bifurcating_prompts.json` : les 27 prompts, avec les listes `ground_truth` et `wrong_indicators` de son
classifieur, extraites sans exécution (analyse `ast`) de `multitoken_hallucination.py`. Les 8 prompts du premier test en
font partie, avec des listes identiques (vérifié). Catégories : 12 fausses prémisses, 8 confabulations, 4 factuels,
2 calculs, 1 question orientée.

## Montage et critères

Identiques à `bassin_akarlar.py` (en-tête du script, commit 0af62ba), sans aucun changement de règle :

- Qwen2.5-1.5B de base, T = 0,7, 15 jetons au plus, classifieur d'Akarlar ;
- jusqu'à 6 paires par prompt, réservoir de 60 trajectoires, 24 graines par paire et par direction ;
- patch persistant aux couches 0, 6, 12, 18, 20, 24, 27 ; échange de jeton ; auto-patch à L20 ;
- nombres aléatoires communs entre conditions ;
- Δ en log-odds, A = Δ_C→H moyen − Δ_H→C moyen, IC95 par bootstrap des paires stratifié par prompt (2000 tirages) ;
- témoins : auto-patch identique ≥ 98 %, échange de jeton avec une borne basse > 0 dans au moins une direction ;
- verdicts : ASYMÉTRIE COMME L'ARTICLE (borne basse de A > 0 et de Δ_C→H > 0), ASYMÉTRIE INVERSE (borne haute de A < 0),
  SYMÉTRIQUE (IC de A dans [−0,5 ; +0,5]), INDÉTERMINÉ sinon ; P2 CONFIRMÉ si la borne basse de Δ_H→C > 0.

Seul changement : `--seed 20261103` et `--prompts data/akarlar_bifurcating_prompts.json`.

## Puissance (calcul fait avant exécution)

En rééchantillonnant 27 prompts parmi les 8 du premier test (8 répétitions, bootstrap de 400 tirages), la demi-largeur
de l'IC de A passe de 0,26 à 0,135 (de 0,11 à 0,17). [hypothèse] Cette estimation suppose que les 27 prompts varient
comme les 8 ; les nouvelles catégories (confabulations, calculs) peuvent varier davantage.

Conséquences, si la demi-largeur vaut environ 0,14 :
- si le vrai A vaut environ −0,25 (valeur du premier test), l'IC exclut 0 : ASYMÉTRIE INVERSE ;
- si le vrai A vaut environ 0 : SYMÉTRIQUE, et un effet de plus de 0,14 en log-odds dans un sens ou dans l'autre est exclu ;
- l'asymétrie annoncée par l'article (corruption facile, correction difficile) demanderait A > 0.

## Prédictions de l'assistant (écrites avant exécution) [spéculation]

ASYMÉTRIE COMME L'ARTICLE 5 %, ASYMÉTRIE INVERSE 35 %, SYMÉTRIQUE 45 %, INDÉTERMINÉ ou NON INTERPRÉTABLE 15 %.
P2 CONFIRMÉ 35 %.

## Limites

- Un seul modèle, celui de l'article. Les 27 prompts sont ceux qu'Akarlar a sélectionnés pour qu'ils bifurquent ; la
  conclusion ne vaut que pour ce type de prompts.
- Certains prompts ont peu de réponses correctes (2 sur 20) : ils donnent moins de paires, et donc moins de poids.

## Commande

    python experiments/bassin_akarlar.py --prompts experiments/data/akarlar_bifurcating_prompts.json \
        --seed 20261103 --out bassin_27.json
