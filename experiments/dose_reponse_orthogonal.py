# experiments/dose_reponse_orthogonal.py
# Suite de dose_reponse.py : contrôles orthogonaux à la direction vraie.
#
# POURQUOI. dose_reponse.py a conclu CONFIRMÉ, mais l'analyse post hoc (dose_reponse_exploratoire.py) a montré que deux de
# ses trois contrôles contiennent une part de la direction vraie d : |cos(permuté, d)| médian 0,42 (max 0,81), |cos(autre
# prompt, d)| médian 0,13 (max 0,92), pour 0,03 attendu d'une direction isotrope en dimension 1024. Ces contrôles ne sont donc
# pas des nuls propres. Ce script les remplace par les MÊMES directions (mêmes graines) privées de leur composante le long de d,
# remises à la norme ‖d‖ : ils portent la même énergie, dans l'espace des données (permuté) ou issue d'un autre bassin
# (autre prompt), mais rien de d.
#
# MONTAGE. Mêmes prompts, même fit (fit.npz), mêmes suites d'évaluation (mêmes lignes, mêmes graines, même t_debut) que
# dose_reponse.py ; la référence est regénérée et ses étiquettes comparées à celles de eval.json (identité attendue).
# Conditions : permute_orth et autre_orth, α ∈ {1 ; 2}. Les résultats de la direction vraie sont ceux de eval.json.
#
# CRITÈRES FIGÉS AVANT TOUTE EXÉCUTION (suite post hoc, décidée après avoir vu les cosinus ; mêmes notations et mêmes seuils que
# dose_reponse.py ; « valides » = les prompts valides de eval.json, ΔNLL(vraie, 1) ≤ 0,5).
#   O-0  Construction. |cos(contrôle orthogonal, d)| < 1e-6 pour tous les prompts.
#   O-A  Efficacité face aux nuls propres. D⊥_p = Δ(vraie, 1) − moyenne des Δ de permute_orth et autre_orth à α = 1 ;
#        médiane des D⊥_p ≥ +0,10 et Wilcoxon exact unilatéral p < 0,05, sur les prompts valides.
#   O-B  Spécificité. |médiane, sur les prompts valides, de la moyenne des Δ des deux contrôles orthogonaux à α = 1| < 0,05.
#   O-C  Dominance. La direction vraie dépasse les DEUX contrôles orthogonaux à α = 1 pour au moins 70 % des prompts valides.
#   VERDICT : CONFIRMÉ si O-0, O-A, O-B et O-C ; RÉFUTÉ si non O-A ; PARTIEL sinon. Il ne remplace pas le verdict de
#   dose_reponse.py : il dit si celui-ci survit à des contrôles propres.
#
# Usage : python experiments/dose_reponse_orthogonal.py --out-dir dr_0p6b [--report-only]

import argparse
import contextlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import dose_reponse as dr  # noqa: E402
from atlas.compat import steer_hook  # noqa: E402

CONTROLES_ORTH = ("permute", "autre")
ALPHAS = (1.0, 2.0)
SEUIL_COS = 1e-6
SEUIL_DOMINANCE = 0.70


def orthogonaliser(v: np.ndarray, d: np.ndarray) -> np.ndarray:
    """v privé de sa composante le long de d, remis à la norme de d."""
    dn = d / np.linalg.norm(d)
    return dr.mettre_a_la_norme(v - (v @ dn) * dn, float(np.linalg.norm(d)))


def cosinus(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluer_prompt(eng, f: dict, info: dict, res: dict) -> dict:
    p, couche = f["idx"], info["couche"]
    cl, pid = f["cl"], list(eng.encoder(dr.PROMPTS[p]))
    rows = [dr.EVAL_OFFSET + p * 256 + j for j in range(res["n_eval"])]

    def generer(vec, alpha):
        out = []
        for b0 in range(0, len(rows), eng.batch_size):
            r = rows[b0:b0 + eng.batch_size]
            ctx = contextlib.nullcontext() if vec is None else steer_hook(eng, couche, vec, alpha=alpha,
                                                                         t_debut=res["t_debut"])
            with ctx:
                out.append(eng.generer_lot([pid] * len(r), r, info["n_steps"], seed_base=dr.SEED_EVAL)["tokens"])
        return np.concatenate(out)

    ref = generer(None, 0.0)
    R0, nll0, _ = dr.rejouer(eng, pid, ref, couche)
    y0 = cl.etiqueter(R0)
    sauve = res["prompts"][str(p)]
    out = {"idx": p, "ref_identique": bool(y0.tolist() == sauve["labels_ref"]), "taux_ref": float(y0.mean()), "conds": {}}
    for nom in CONTROLES_ORTH:
        v = orthogonaliser(f["controles"][nom], f["d"])
        cos = cosinus(v, f["d"])
        for a in ALPHAS:
            toks = generer(v, a)
            R1, nll1, _ = dr.rejouer(eng, pid, toks, couche)
            y1 = cl.etiqueter(R1)
            out["conds"][f"{nom}_orth@{a:g}"] = {
                "direction": f"{nom}_orth", "alpha": a, "cos_avec_d": cos, "taux": float(y1.mean()),
                "delta": float(y1.mean() - y0.mean()), "delta_nll": float(nll1.mean() - nll0.mean()),
                "frac_modifiees": float((toks != ref).any(1).mean()), "labels": y1.tolist()}
    return out


def phase(eng, args, info, donnees, res):
    out = Path(args.out_dir)
    chemin = out / "eval_orth.json"
    fits = dr.preparer(info, donnees, res["prompts_selectionnes"], Namespace(min_rare=args.min_rare))
    sortie = {"prompts_selectionnes": [f["idx"] for f in fits], "prompts": {}}
    if chemin.exists():
        prev = json.loads(chemin.read_text())
        if prev.get("prompts_selectionnes") == sortie["prompts_selectionnes"]:
            sortie["prompts"] = prev["prompts"]
    for f in fits:
        if str(f["idx"]) in sortie["prompts"]:
            continue
        t0 = time.time()
        sortie["prompts"][str(f["idx"])] = evaluer_prompt(eng, f, info, res)
        chemin.write_text(json.dumps(sortie, indent=1, ensure_ascii=False))
        c = sortie["prompts"][str(f["idx"])]["conds"]
        print(f"[orth] prompt {f['idx']:2d} réf identique : {sortie['prompts'][str(f['idx'])]['ref_identique']} ; "
              f"permute⊥@1 {c['permute_orth@1']['delta']:+.2f}, autre⊥@1 {c['autre_orth@1']['delta']:+.2f} ; "
              f"{time.time() - t0:.0f} s", flush=True)
    chemin.write_text(json.dumps(sortie, indent=1, ensure_ascii=False))
    return sortie


def verdict_orth(res: dict, orth: dict) -> dict:
    valides = [p for p in res["prompts"].values() if p["conds"]["vraie@1"]["delta_nll"] <= dr.SEUIL_FLUIDITE
               and str(p["idx"]) in orth["prompts"]]
    if len(valides) < dr.MIN_VALIDES:
        return {"verdict": "INDÉTERMINÉ", "n_valides": len(valides)}
    D, moy, bat = [], [], []
    cosmax = 0.0
    for p in valides:
        o = orth["prompts"][str(p["idx"])]["conds"]
        ds = [o[f"{c}_orth@1"]["delta"] for c in CONTROLES_ORTH]
        v = p["conds"]["vraie@1"]["delta"]
        moy.append(float(np.mean(ds)))
        D.append(v - moy[-1])
        bat.append(all(v > x for x in ds))
        cosmax = max(cosmax, max(abs(o[f"{c}_orth@1"]["cos_avec_d"]) for c in CONTROLES_ORTH))
    o0 = cosmax < SEUIL_COS
    pw = dr.wilcoxon_exact(D)
    oa = float(np.median(D)) >= dr.SEUIL_D and pw < dr.SEUIL_P
    ob = abs(float(np.median(moy))) < dr.SEUIL_CTRL
    oc = float(np.mean(bat)) >= SEUIL_DOMINANCE
    return {"n_valides": len(valides), "O_0": o0, "O_A": oa, "O_B": ob, "O_C": oc, "cos_max": cosmax,
            "mediane_D": float(np.median(D)), "p_wilcoxon": pw, "positifs": int(sum(x > 0 for x in D)),
            "mediane_controles": float(np.median(moy)), "part_dominance": float(np.mean(bat)),
            "verdict": "CONFIRMÉ" if (o0 and oa and ob and oc) else ("RÉFUTÉ" if not oa else "PARTIEL")}


def rapport(res: dict, orth: dict) -> str:
    v = verdict_orth(res, orth)
    L = [f"=== CONTRÔLES ORTHOGONAUX ({res['fit']['modele']}, couche {res['fit']['couche']}) ===",
         f"Référence regénérée identique à eval.json : {sum(o['ref_identique'] for o in orth['prompts'].values())}"
         f"/{len(orth['prompts'])} prompts", "",
         f"{'prompt':>6} {'vraie@1':>8} {'perm⊥@1':>8} {'autre⊥@1':>8}   {'perm@1':>7} {'autre@1':>7} {'(anciens)':>9} "
         f"{'vraie@2':>8} {'perm⊥@2':>8} {'autre⊥@2':>8} {'ΔNLL perm⊥/autre⊥ @1':>22}"]
    for p in res["prompts"].values():
        if str(p["idx"]) not in orth["prompts"]:
            continue
        o = orth["prompts"][str(p["idx"])]["conds"]
        c = p["conds"]
        L.append(f"{p['idx']:>6} {c['vraie@1']['delta']:>+8.2f} {o['permute_orth@1']['delta']:>+8.2f} "
                 f"{o['autre_orth@1']['delta']:>+8.2f}   {c['permute@1']['delta']:>+7.2f} {c['autre@1']['delta']:>+7.2f} "
                 f"{'':>9} {c['vraie@2']['delta']:>+8.2f} {o['permute_orth@2']['delta']:>+8.2f} "
                 f"{o['autre_orth@2']['delta']:>+8.2f} "
                 f"{o['permute_orth@1']['delta_nll']:>+10.2f}/{o['autre_orth@1']['delta_nll']:>+.2f}")
    L.append("")
    if v["verdict"] == "INDÉTERMINÉ":
        L.append(f"Prompts valides : {v['n_valides']} (< {dr.MIN_VALIDES})")
    else:
        L.append(f"O-0 construction : |cos| max = {v['cos_max']:.1e} (< {SEUIL_COS:.0e}) → {'oui' if v['O_0'] else 'non'}")
        L.append(f"O-A efficacité : médiane D⊥ = {v['mediane_D']:+.3f} (≥ {dr.SEUIL_D}), {v['positifs']}/{v['n_valides']} "
                 f"positifs, Wilcoxon exact p = {v['p_wilcoxon']:.2e} (< {dr.SEUIL_P}) → {'oui' if v['O_A'] else 'non'}")
        L.append(f"O-B spécificité : médiane des Δ des contrôles orthogonaux = {v['mediane_controles']:+.3f} "
                 f"(|.| < {dr.SEUIL_CTRL}) → {'oui' if v['O_B'] else 'non'}")
        L.append(f"O-C dominance : vraie > les deux contrôles pour {v['part_dominance']:.0%} des prompts valides "
                 f"(≥ {SEUIL_DOMINANCE:.0%}) → {'oui' if v['O_C'] else 'non'}")
    L.append(f"VERDICT (contrôles orthogonaux) : {v['verdict']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-rare", type=int, default=15)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    res = json.loads((out / "eval.json").read_text())
    if not args.report_only:
        info, donnees = dr.charger(args.out_dir)
        args.n_steps = info["n_steps"]
        eng, _ = dr.construire(args, info)
        orth = phase(eng, args, info, donnees, res)
    else:
        orth = json.loads((out / "eval_orth.json").read_text())
    print(rapport(res, orth))


if __name__ == "__main__":
    main()
