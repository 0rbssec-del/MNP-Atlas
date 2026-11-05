# experiments/echantillon_classifieur_v2.py
# Validation du classifieur, v2 : nouvel échantillon de 160 suites (mêmes strates que v1, graine 20261108), sans les
# 160 suites de v1, plus 20 suites de v1 réinsérées à l'aveugle (contrôle de l'effet des nouvelles règles).
# Plan : experiments/VALIDATION_CLASSIFIEUR.md, amendement 1.
import collections, gzip, json, random
from pathlib import Path

D = Path(__file__).resolve().parent / "resultats" / "validation_classifieur"
V2 = D / "v2"; V2.mkdir(exist_ok=True)
a = json.loads(gzip.open(D / "bassin_confirmation_textes.json.gz").read())
rows, P = a["lignes"], a["textes_prompts"]
cle_v1 = json.loads((D / "cle_cachee.json").read_text())
K = lambda r: (r["idx"], r["pair"], r["dir"], r["cond"], r["seed"])  # noqa: E731
deja = {K(r) for r in cle_v1}
avec = {(r["idx"], r["pair"], r["dir"]) for r in rows if r["meme"]}
rng = random.Random(20261108)
ech = []
for cond in ("base", "echange", "croise_L24", "meme_L24"):
    for d in ("C->H", "H->C"):
        cand = [r for r in rows if r["cond"] == cond and r["dir"] == d and (r["idx"], r["pair"], d) in avec
                and K(r) not in deja]
        rng.shuffle(cand); par = collections.Counter(); pick = []
        for r in cand:
            k = (r["idx"], r["pair"])
            if par[k] < 2:
                pick.append(dict(r, origine="v2")); par[k] += 1
            if len(pick) == 20:
                break
        ech += pick
index = {K(r): r for r in rows}
reprises = rng.sample(cle_v1, 20)
ech += [dict(index[K(c)], origine="v1", n_v1=c["n"]) for c in reprises]
rng.shuffle(ech)
items, cle = [], []
for i, r in enumerate(ech, 1):
    items.append({"n": i, "idx": r["idx"], "prompt": P[str(r["idx"])], "suite": r["texte"]})
    cle.append({"n": i, "origine": r["origine"], "n_v1": r.get("n_v1"),
                **{k: r[k] for k in ("idx", "pair", "dir", "cond", "seed", "label", "strict")}})
(V2 / "a_etiqueter.json").write_text(json.dumps(items, ensure_ascii=False, indent=0))
(V2 / "cle_cachee.json").write_text(json.dumps(cle, ensure_ascii=False))
print(len(items), "suites,", sum(c["origine"] == "v1" for c in cle), "reprises de v1,",
      len({x["idx"] for x in items}), "prompts")
