# experiments/analyse_gsm8k.py
# Analyses fixées d'avance (experiments/AUDIT_GSM8K.md) sur les étiquettes à l'aveugle d'Erick.
# A1 erreur des extracteurs par strate ; A2 exactitude corrigée (estimateur stratifié du plan) ; A3 écarts entre modèles.
# Secondaire, non pré-enregistré : estimateur « recensement » (strate D entièrement étiquetée, verdict automatique en
# strate A, où aucune erreur n'a été trouvée), apparié par problème.
import gzip, json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent / "resultats" / "audit_gsm8k"
res = json.loads(gzip.open(D / "audit_gsm8k.json.gz").read())
lab = dict(x.split(":") for x in (D / "etiquettes_erick.txt").read_text().strip().split(","))
cle = json.loads((D / "cle_cachee.json").read_text())
M = list(res["modeles"]); NTOT = sum(len(res["modeles"][m]["lignes"]) for m in M)
it = cle["items"]
for x in it:
    x["juste"] = lab[str(x["n"])] == "J"          # AUCUNE compte comme fausse
rng = np.random.default_rng(20261107)
B = 10000
out = {}

def cp_haut(k, n, a=0.05):                         # borne haute de Clopper-Pearson (bilatérale)
    from scipy.stats import beta
    return 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))

# A1
taille = {s: sum(cle["taille_strates"][m][s] for m in M) for s in "DA"}
out["A1"] = {}
for e in ["strict", "flexible"]:
    r = {}
    for s in "DA":
        L = [x for x in it if x["strate"] == s]
        k = sum(x["ok_" + e] != x["juste"] for x in L)
        r[s] = {"erreurs": k, "n": len(L), "taux": k / len(L), "ic95_haut": cp_haut(k, len(L))}
    est = sum(r[s]["taux"] * taille[s] for s in "DA") / NTOT
    haut = sum(r[s]["ic95_haut"] * taille[s] for s in "DA") / NTOT
    r["global_stratifie"], r["global_borne_haute"] = est, haut
    out["A1"][e] = r
    print(f"[A1] {e} : D {r['D']['erreurs']}/{r['D']['n']}, A {r['A']['erreurs']}/{r['A']['n']} "
          f"(borne haute {r['A']['ic95_haut']:.3f}) ; global stratifié {est:.4f} (borne haute {haut:.4f})")

# A2 : estimateur du plan, par modèle : somme sur strates de N_s,m/500 × taux de JUSTE des étiquetés (s, m)
def cellules(m):
    return {s: np.array([x["juste"] for x in it if x["strate"] == s and x["modele"] == m], float) for s in "DA"}

def acc_plan(m, tirage=None):
    c, N = cellules(m), cle["taille_strates"][m]
    tot = 0.0
    for s in "DA":
        if N[s] == 0:
            continue
        v = c[s] if tirage is None else c[s][tirage[(m, s)]]
        tot += N[s] * v.mean()
    return tot / sum(N.values())

tir = [{(m, s): rng.integers(0, len(cellules(m)[s]), len(cellules(m)[s])) for m in M for s in "DA"
        if len(cellules(m)[s])} for _ in range(B)]
out["A2"] = {}
for m in M:
    bs = np.array([acc_plan(m, t) for t in tir])
    c = cellules(m)
    out["A2"][m] = {"acc": acc_plan(m), "ic95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                    "n_etiquetes": {s: int(len(c[s])) for s in "DA"}}
    print(f"[A2] {m} : corrigée (plan) {acc_plan(m):.3f} [{out['A2'][m]['ic95'][0]:.3f} ; {out['A2'][m]['ic95'][1]:.3f}] "
          f"(étiquetés D {len(c['D'])}, A {len(c['A'])})")

# Secondaire : verdict corrigé par réponse (D : étiquette ; A : verdict commun des extracteurs)
lab_d = {(x["modele"], x["i"]): x["juste"] for x in it if x["strate"] == "D"}
V = {}
for m in M:
    L = res["modeles"][m]["lignes"]
    V[(m, "strict")] = np.array([x["ok_strict"] for x in L], float)
    V[(m, "flexible")] = np.array([x["ok_flexible"] for x in L], float)
    V[(m, "recensement")] = np.array([lab_d.get((m, x["i"]), x["ok_strict"]) if x["ok_strict"] != x["ok_flexible"]
                                      else x["ok_strict"] for x in L], float)
n = len(res["modeles"][M[0]]["lignes"]); Bi = rng.integers(0, n, (B, n))
print("[SECONDAIRE] recensement :", {m.split('/')[1]: round(V[(m, 'recensement')].mean(), 3) for m in M})

# A3 : écarts entre paires
out["A3"] = {}
for a in range(3):
    for b in range(a + 1, 3):
        p = f"{M[a].split('/')[1]} − {M[b].split('/')[1]}"; r = {}
        for k in ["strict", "flexible", "recensement"]:
            d = V[(M[a], k)] - V[(M[b], k)]; bs = d[Bi].mean(1)
            r[k] = [float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        bs = np.array([acc_plan(M[a], t) - acc_plan(M[b], t) for t in tir])
        r["plan"] = [acc_plan(M[a]) - acc_plan(M[b]), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        out["A3"][p] = r
        print(f"[A3] {p} : " + " ; ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f} ; {v[2]:+.3f}]" for k, v in r.items()))

# Verdict selon les critères du plan (correction = estimateur du plan)
fiable_err = all(out["A1"][e]["global_stratifie"] <= 0.02 for e in ["strict", "flexible"])
inv, inv_sig, ecart5 = [], False, False
for p, r in out["A3"].items():
    for e in ["strict", "flexible"]:
        if np.sign(r[e][0]) != np.sign(r["plan"][0]) and r[e][0] != 0:
            inv.append((p, e)); inv_sig |= (r["plan"][1] > 0 or r["plan"][2] < 0)
        ecart5 |= abs(r[e][0] - r["plan"][0]) >= 0.05
if inv and inv_sig or ecart5:
    v = "CLASSEMENT DÉPENDANT DU CORRECTEUR"
elif fiable_err and not inv:
    v = "CORRECTEUR FIABLE"
elif not fiable_err and not inv:
    v = "ERREUR SANS EFFET SUR LE CLASSEMENT"
else:
    v = "INDÉTERMINÉ"
out["verdict"] = {"verdict": v, "inversions": inv, "inversion_significative": bool(inv_sig), "ecart_5_points": bool(ecart5),
                  "erreur_au_plus_2pc": bool(fiable_err)}
print(f"[VERDICT] {v} ; inversions {inv} ; significative {inv_sig} ; écart ≥ 5 points {ecart5}")
(D / "analyse.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
