# Témoin de relance de Qwen2.5-1.5B : résultat (5 nov. 2026)

Plan : `experiments/AUDIT_GSM8K.md`, amendement 3 (commit 1f1d658). Sorties : `relance_gsm8k.json.gz`, journal
`log_relance_gsm8k.txt`.

## Verdict pré-enregistré : COMPOSITION DU LOT. Mais l'échec du témoin était surtout un artefact de comparaison.

| Comparaison | Chaînes brutes | Sans jetons `<|endoftext|>` de fin |
|---|---|---|
| R1a = R1b (mêmes lots) | 64/64 | 64/64 |
| R2 = R1a, lot commun | 32/32 | 32/32 |
| R2 = R1a, lot modifié | 0/18 | **17/18** |
| R3 (lots de 1) = passe principale | 15/200 | **193/200** |
| float32, lot modifié | 0/18 | 18/18 |
| float32 = float16 (R1a) | 63/64 | 63/64 |

**Erreur de l'assistant.** Le témoin comparait les chaînes décodées avec `skip_special_tokens=False`. Dans un lot, une
suite déjà finie est complétée par des `<|endoftext|>` jusqu'à la longueur de la plus longue suite du lot, et ce nombre
dépend de la composition du lot. Qwen2.5-1.5B finit par ce jeton (462 réponses sur 500), SmolLM3 jamais : d'où 64 %
contre 100 %. Sans ces jetons, le témoin de l'étape 1 aurait été d'environ 49/50 [estimation : 17/18 du lot modifié
+ 32/32 du lot commun].

**Effet réel de la composition du lot, après correction** : 1 réponse sur 18 en float16, 0 sur 18 en float32 ; 7 réponses
sur 200 changent entre lots de 1 et lots de 32, sans qu'aucun verdict ne change (exactitude 0,575 dans les deux cas,
strict et flexible, différence 0). Les sorties de l'étape 1, l'étiquetage et `RESULTAT.md` ne sont pas touchés : les
jetons de fin ne contiennent aucun chiffre.
