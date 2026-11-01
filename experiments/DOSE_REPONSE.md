# Couche 3 : dose-réponse du steering sur les bassins (Qwen3-0.6B-Base)

Script : `experiments/dose_reponse.py` (critères et verdict écrits en en-tête, commit `5a30f2b`, avant toute évaluation).
Phase fit : commit `5f42131`. Données : `experiments/resultats/dose_reponse_0p6b/` (`fit.json`, `eval.json`,
`exploratoire.txt` ; `fit.npz`, 38 Mo, non versionné). Analyses post hoc : `experiments/dose_reponse_exploratoire.py`.

## Conclusion

Sur les 19 prompts multimodaux de B2 (Qwen3-0.6B-Base, couche 17, bf16), ajouter à chaque pas de génération la direction
μ_rare − μ_dominant **fait monter la part du régime rare** dans des suites neuves, de façon dose-dépendante et sans
dégrader le texte à α = 1. Les quatre critères pré-enregistrés sont satisfaits : **verdict CONFIRMÉ**. Les régimes
concernés sont des **formats de texte** (questionnaire à choix multiples ou « Réponse : None » contre prose), pas des
contenus sémantiques. Deux des trois contrôles d'origine contenaient une part de la direction vraie ; refaits **orthogonaux à
d** (suite pré-enregistrée, `dose_reponse_orthogonal.py`), ils donnent aussi **CONFIRMÉ** (voir « Contrôles orthogonaux »).
À α = 2, le texte se dégrade et les perturbations hors direction déplacent aussi le taux : la spécificité tient à α = 1.

## Montage

Fit : 192 suites × 32 jetons par prompt (identifiants de bruit p·256 + j, seed_base = SEED + 17), R_h = moyenne des états de
la couche 17 sur les jetons générés (rejeu non perturbé), classifieur des deux régimes figé (standardisation, ACP à 10
composantes, centres des 2-moyennes), direction d = μ_rare − μ_dominant. Évaluation : 96 suites **neuves** par condition
(seed_base = SEED + 99), mêmes lignes et mêmes graines que la référence (comparaison appariée). L'effet est lu en rejouant
les jetons produits dans le modèle **non perturbé** (régime par le classifieur figé ; NLL par jeton comme garde de
fluidité). Contrôles de même norme ‖d‖ : direction gaussienne, directions d'étiquettes permutées, direction vraie d'un autre
prompt. α = 1 ajoute la différence complète des centres à chaque position ; α = 0 est l'identité (19 prompts sur 19).

## Critères pré-enregistrés (unité : le prompt ; 19 évalués, 18 valides)

| Critère | Seuil | Résultat | |
|---|---|---|---|
| P-A efficacité | médiane D ≥ +0,10 et Wilcoxon exact p < 0,05 | médiane D = +0,299, p = 3,8·10⁻⁵ (17 prompts valides sur 18 positifs) | oui |
| P-B dose-réponse | ρ(α, taux) > 0 pour ≥ 70 % des prompts valides | 89 % | oui |
| P-C fluidité | ≥ 50 % des prompts valides à α = 1 | 95 % (18/19 ; le prompt 38 : ΔNLL +0,55) | oui |
| P-D spécificité | \|médiane des Δ de contrôle\| < 0,05 à α = 1 | +0,005 | oui |

Δ(vraie, α = 1) : médiane +0,27 pour un taux de référence médian de 0,39.

## Analyses post hoc (non pré-enregistrées)

- **Signe cohérent** : Δ(α = +1) > 0 pour 18/19 prompts, Δ(α = −1) < 0 pour 19/19 ; les deux pour 18/19. Le prompt 0 est
  l'exception : à α = +1 le taux baisse (−0,23).
- **Contrôles non indépendants de d** : |cos(contrôle, d)| médian 0,03 pour le gaussien (isotrope en dimension 1024 : 0,031),
  **0,42 pour les étiquettes permutées (max 0,81)** et 0,13 pour l'autre prompt (max 0,92). L'effet du contrôle permuté suit
  son alignement avec d (corrélation signée +0,80). La direction vraie dépasse **tous** les contrôles dans 14 prompts valides
  sur 18 seulement.
- **Robustesse de P-A** avec le seul contrôle gaussien : médiane D = +0,260, 16/18 positifs, p = 1,9·10⁻⁵.
- **Dégradation** : ΔNLL moyen +0,13 à α = 1 (max +0,55) mais **+0,54 à α = 2** (9 prompts sur 19 au-dessus de 0,5). La
  dose utile est donc étroite ; les non-monotonies (prompts 19, 51, 38) apparaissent aux doses fortes.
- **Nature des régimes** (suites de fit) : prompt 47, rare = questionnaire (« which of the following countries? A. … B. … »)
  contre prose déclarative ; prompt 18, rare = prose de recette contre format quiz (« A. Vrai B. Faux Answer: ») ; prompt 0,
  rare = « Réponse: None » contre prose.

## Contrôles orthogonaux (suite, critères figés avant exécution : commit `31452f5`)

Les contrôles « étiquettes permutées » et « autre prompt » de l'expérience principale, mêmes graines, privés de leur
composante le long de d et remis à la norme ‖d‖ ; mêmes suites d'évaluation (référence regénérée identique à `eval.json` :
19 prompts sur 19). Données : `eval_orth.json`, `orthogonal.txt`.

| Critère | Seuil | Résultat | |
|---|---|---|---|
| O-0 construction | \|cos\| < 1e-6 | 2,8·10⁻¹⁶ | oui |
| O-A efficacité | médiane D⊥ ≥ +0,10, Wilcoxon p < 0,05 | médiane +0,312, 17/18 positifs, p = 1,1·10⁻⁵ | oui |
| O-B spécificité | \|médiane des contrôles\| < 0,05 | −0,018 | oui |
| O-C dominance | vraie > les deux contrôles pour ≥ 70 % des prompts valides | 89 % | oui |

**Verdict : CONFIRMÉ.** Post hoc, entre prompts, à α = 1 : Δ vraie +0,275 (écart-type 0,175) ; permuté⊥ −0,041 (0,130) ;
autre⊥ +0,019 (0,129) ; gaussien +0,008 (0,107). Les contrôles ont une dispersion supérieure au bruit d'échantillonnage
attendu d'un Δ apparié (≈ 0,07) : pris un à un, ils déplacent parfois le taux de ±0,2 avec un signe aléatoire, donc un contrôle
isolé ne prouve rien sur un prompt ; c'est l'agrégat sur 18 prompts qui porte la conclusion. À α = 2 (ΔNLL moyen +0,5, texte
dégradé), les contrôles orthogonaux ont un Δ moyen négatif (−0,08 et −0,06) et 9 à 13 prompts sur 19 au-delà de ±0,14 : les
fortes perturbations hors direction déplacent aussi le taux.

## Limites

Un seul modèle, une seule couche (17), une seule date de départ (t_debut = 0), bf16 sur CPU, 96 suites par condition. Les
régimes sont des formats de texte : « actionnable » veut dire que l'ajout de la direction fait basculer le format, pas
qu'on pilote un contenu. Prompts sélectionnés par le test de B2 sur nos propres suites de fit (pas par la liste
publiée de B2, qui concerne d'autres modèles). L'interprétation causale se limite à : à ce modèle et cette couche, ajouter
cette direction déplace ces suites vers ce régime plus que des directions de même norme.
