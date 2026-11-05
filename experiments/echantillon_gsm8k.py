# experiments/echantillon_gsm8k.py
# Tirage de l'échantillon à étiqueter à l'aveugle (plan : experiments/AUDIT_GSM8K.md, section « Étiquetage à l'aveugle »).
# Strate D : réponses où strict et flexible rendent des verdicts différents (jusqu'à 90, parts égales entre modèles
# autant que possible). Strate A : complément jusqu'à 150, tiré au hasard parmi les réponses en accord. Graine 20261106.
# Sorties : a_etiqueter.json (ce que voit l'étiqueteur, sans modèle ni verdict) et cle_cachee.json (correspondance).
import gzip, json, sys
from pathlib import Path
import numpy as np

SEED, TOTAL, MAX_D = 20261106, 150, 90
src = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/resultats/audit_gsm8k/audit_gsm8k.json.gz")
out = src.parent
res = json.loads(gzip.open(src).read())
test = json.loads(Path("experiments/data/gsm8k_test.json").read_text())
rng = np.random.default_rng(SEED)

D, A = {}, []
for m, d in res["modeles"].items():
    for x in d["lignes"]:
        (D.setdefault(m, []) if x["ok_strict"] != x["ok_flexible"] else A).append((m, x))
# Strate D : tourniquet entre modèles jusqu'à MAX_D
pris_d = []
files = {m: [D[m][k] for k in rng.permutation(len(D[m]))] for m in sorted(D)}
while len(pris_d) < MAX_D and any(files.values()):
    for m in sorted(files):
        if files[m] and len(pris_d) < MAX_D:
            pris_d.append(files[m].pop(0))
pris_a = [A[k] for k in rng.choice(len(A), size=TOTAL - len(pris_d), replace=False)]
items = [("D", m, x) for m, x in pris_d] + [("A", m, x) for m, x in pris_a]
ordre = rng.permutation(len(items))
vue, cle = [], []
for n, k in enumerate(ordre, 1):
    s, m, x = items[k]
    vue.append({"n": n, "question": test[x["i"]]["question"],
                "reponse": x["reponse"].replace("<|endoftext|>", "").rstrip(),
                "attendu": x["or"].split("####")[-1].strip()})
    cle.append({"n": n, "strate": s, "modele": m, "i": x["i"], "strict": x["strict"], "flexible": x["flexible"],
                "ok_strict": x["ok_strict"], "ok_flexible": x["ok_flexible"]})
taille = {m: {"D": len(D.get(m, [])), "A": sum(1 for mm, _ in A if mm == m)} for m in res["modeles"]}
(out / "a_etiqueter.json").write_text(json.dumps(vue, ensure_ascii=False, indent=1))
(out / "cle_cachee.json").write_text(json.dumps({"seed": SEED, "taille_strates": taille, "items": cle}, ensure_ascii=False, indent=1))
print("strates (population) :", taille)
print(f"échantillon : D {len(pris_d)}, A {len(pris_a)}")
