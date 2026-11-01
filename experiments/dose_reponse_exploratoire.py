# experiments/dose_reponse_exploratoire.py
# Analyses POST HOC de la dose-réponse (écrites après avoir vu le verdict de dose_reponse.py ; non pré-enregistrées, donc
# exploratoires : elles qualifient le verdict, elles ne le remplacent pas).
#   A. Dégradation du texte (ΔNLL) selon la dose et la direction.
#   B. Les contrôles « permuté » et « autre prompt » sont-ils indépendants de la direction vraie ? (cosinus)
#   C. Robustesse de P-A avec le seul contrôle isotrope (direction gaussienne) ; cohérence du signe (α = +1 / −1).
#   D. Exemples de suites de chaque régime (jetons décodés).
#
# Usage : python experiments/dose_reponse_exploratoire.py --out-dir dr_0p6b [--textes 47 18 0]

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import dose_reponse as dr  # noqa: E402


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--textes", nargs="*", type=int, default=[47, 18, 0])
    args = ap.parse_args()
    out = Path(args.out_dir)
    res = json.loads((out / "eval.json").read_text())
    fit = np.load(out / "fit.npz")
    P = res["prompts"]
    idx = res["prompts_selectionnes"]

    print("=== A. Dégradation du texte : ΔNLL moyen (max) par condition, en nat par jeton ===")
    for k in ["vraie@0.5", "vraie@1", "vraie@2", "vraie@-1", "aleatoire@1", "aleatoire@2", "permute@1", "permute@2",
              "autre@1", "autre@2"]:
        v = [p["conds"][k]["delta_nll"] for p in P.values() if k in p["conds"]]
        print(f"  {k:12s} {np.mean(v):+.2f} ({np.max(v):+.2f})")
    print("  prompts à ΔNLL > 0,5 pour la direction vraie : α = 1 :",
          [p["idx"] for p in P.values() if p["conds"]["vraie@1"]["delta_nll"] > dr.SEUIL_FLUIDITE], "; α = 2 :",
          [p["idx"] for p in P.values() if p["conds"]["vraie@2"]["delta_nll"] > dr.SEUIL_FLUIDITE])

    print("\n=== B. Indépendance des contrôles vis-à-vis de la direction vraie d ===")
    fits = {}
    for p in idx:
        R = np.asarray(fit[f"R{p}"], np.float64)
        cl, lab = dr.Classifieur.ajuster(R, seed=dr.SEED)
        fits[p] = dict(R=R, lab=lab, cl=cl, d=dr.direction_vraie(R, lab, cl.rare))
    c = {k: [] for k in dr.CONTROLES}
    for k, p in enumerate(idx):
        f = fits[p]
        ct = dr.controles(f["R"], f["lab"], f["cl"].rare, fits[idx[(k + 1) % len(idx)]]["d"], dr.SEED + 1000 + p)
        for nom in dr.CONTROLES:
            c[nom].append(cos(ct[nom], f["d"]))
    dim = fits[idx[0]]["R"].shape[1]
    print(f"  |cos(contrôle, d)| médian [max] ; isotrope en dimension {dim} : ≈ {1 / np.sqrt(dim):.3f}")
    for nom in dr.CONTROLES:
        a = np.abs(c[nom])
        eff = [P[str(p)]["conds"][f"{nom}@1"]["delta"] for p in idx]
        print(f"  {nom:9s} {np.median(a):.3f} [{a.max():.3f}] ; corr(cos signé, Δ du contrôle à α = 1) = "
              f"{np.corrcoef(c[nom], eff)[0, 1]:+.2f}")

    print("\n=== C. Robustesse de P-A et cohérence du signe ===")
    valides = [p for p in P.values() if p["conds"]["vraie@1"]["delta_nll"] <= dr.SEUIL_FLUIDITE]
    for nom, cles in (("trois contrôles (P-A officiel)", dr.CONTROLES), ("gaussien seul (isotrope)", ("aleatoire",))):
        D = [p["conds"]["vraie@1"]["delta"] - np.mean([p["conds"][f"{k}@1"]["delta"] for k in cles]) for p in valides]
        print(f"  D avec {nom} : médiane {np.median(D):+.3f}, positifs {sum(x > 0 for x in D)}/{len(D)}, "
              f"Wilcoxon exact p = {dr.wilcoxon_exact(D):.2e}")
    meilleur = [p["conds"]["vraie@1"]["delta"] > max(p["conds"][f"{k}@1"]["delta"] for k in dr.CONTROLES) for p in valides]
    print(f"  direction vraie > TOUS les contrôles à α = 1 : {sum(meilleur)}/{len(valides)} prompts valides")
    plus = sum(p["conds"]["vraie@1"]["delta"] > 0 for p in P.values())
    moins = sum(p["conds"]["vraie@-1"]["delta"] < 0 for p in P.values())
    both = sum(p["conds"]["vraie@1"]["delta"] > 0 and p["conds"]["vraie@-1"]["delta"] < 0 for p in P.values())
    print(f"  Δ(α = +1) > 0 : {plus}/{len(P)} ; Δ(α = −1) < 0 : {moins}/{len(P)} ; les deux (signe cohérent) : {both}/{len(P)}")
    print(f"  Δ vraie@1 : médiane {np.median([p['conds']['vraie@1']['delta'] for p in P.values()]):+.2f} ; "
          f"taux de référence médian {np.median([p['taux_ref'] for p in P.values()]):.2f}")

    print("\n=== D. Exemples de suites de fit (jetons décodés) ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(res["fit"]["modele"])
    for p in args.textes:
        if p not in fits:
            continue
        f = fits[p]
        tk = fit[f"tok{p}"]
        rare = f["cl"].rare
        print(f"--- prompt {p} : « {dr.PROMPTS[p]} » (fit : {int((f['lab'] == rare).sum())} rares / {len(f['lab'])})")
        for nom, sel in (("RARE", np.where(f["lab"] == rare)[0][:3]), ("DOMINANT", np.where(f["lab"] != rare)[0][:3])):
            for i in sel:
                print(f"  {nom:8s}| " + tok.decode(tk[i].tolist()).replace("\n", "⏎")[:110])


if __name__ == "__main__":
    main()
