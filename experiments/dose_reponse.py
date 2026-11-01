# experiments/dose_reponse.py
# Couche 3 – test dose-réponse du steering sur les bassins (régimes de B2).
#
# QUESTION. Un régime détecté par B2 (bimodalité des états cachés d'un prompt) est-il causalement actionnable ? Ajouter
# la direction μ_rare − μ_dominant à la couche moyenne, à chaque pas de génération, augmente-t-il la part du régime rare
# dans les suites, de façon dose-dépendante, plus que des directions de même norme sans lien avec le régime, et sans
# dégrader le texte ?
#
# MONTAGE (protocole « sgz » du moteur v0.2 : Gumbel à nombres aléatoires communs, T = 0,6, prompt brut).
#   fit  : par prompt, N_FIT suites de N_STEPS jetons (identifiants de bruit p·256 + j, seed_base = SEED + 17, comme B2) ;
#          R_h = moyenne des états de la couche sur les jetons générés (passe rejouée, non perturbée) ; test de bimodalité
#          de B2 (analyser_regimes) ; classifieur des deux régimes figé sur ces suites (standardisation, ACP à 10
#          composantes, centres des 2-moyennes) ; régime rare = le plus petit ; direction d = μ_rare − μ_dominant.
#   eval : sur des suites NEUVES (autres identifiants de bruit, seed_base = SEED + 99), jamais utilisées pour ajuster la
#          direction ni le classifieur. Pour chaque condition, MÊMES lignes et MÊMES graines que la référence
#          (comparaison appariée) : la suite change seulement si l'ajout fait changer un jeton. L'effet est mesuré en
#          REJOUANT les jetons produits dans le modèle NON perturbé : part classée « rare » par le classifieur figé, et NLL
#          par jeton (garde de fluidité). Rien n'est lu sur les états perturbés (pas de circularité).
#   Conditions : référence (α = 0) ; direction vraie à α ∈ {0,5 ; 1 ; 2 ; −1} ; trois contrôles de MÊME norme ‖d‖ à
#   α ∈ {1 ; 2} : direction aléatoire, direction d'étiquettes permutées (même tailles de groupes), direction vraie d'un
#   autre prompt. α = 1 ajoute à chaque position la différence complète des centres ; α = 0 est l'identité (vérifiée).
#
# CRITÈRES FIGÉS AVANT TOUTE EXÉCUTION DE LA PHASE « eval » (unité statistique : le prompt).
#   Notation : taux(c) = part de suites neuves classées « rare » ; Δ(c) = taux(c) − taux(référence) ; ΔNLL(c) = NLL moyenne
#   par jeton (modèle non perturbé) de la condition moins celle de la référence. Une condition est « fluide » si ΔNLL ≤ 0,5
#   nat/jeton. Un prompt est « valide » si la direction vraie à α = 1 est fluide.
#   P-A  Efficacité. Parmi les prompts valides, D_p = Δ(vraie, 1) − moyenne des Δ des trois contrôles à α = 1.
#        Satisfait si au moins 6 prompts valides, médiane des D_p ≥ +0,10 et test des rangs signés de Wilcoxon exact
#        unilatéral (D_p > 0) p < 0,05.
#   P-B  Dose-réponse. Pour au moins 70 % des prompts valides, le coefficient de Spearman entre α ∈ {0 ; 0,5 ; 1 ; 2} et le
#        taux de la direction vraie est > 0.
#   P-C  Fluidité. Au moins 50 % de tous les prompts évalués sont valides.
#   P-D  Spécificité. Médiane, sur les prompts valides, de la moyenne des Δ des contrôles à α = 1 : |médiane| < 0,05.
#   VERDICT : INDÉTERMINÉ si moins de 6 prompts valides ; sinon CONFIRMÉ si P-A, P-B, P-C et P-D ; RÉFUTÉ si non P-A ;
#   PARTIEL dans les autres cas.
#   Exploratoire (non décisif) : α = −1 (Δ < 0 attendu), α = 2, taux de suites modifiées, IC95 apparié par prompt.
#   Prompts : ceux que B2 déclare multimodaux (analyser_regimes sur les 64 prompts du fit, correction BH) ou, avec
#   --selection b2, la liste publiée de B2 si le test de bimodalité est aussi < 0,05 sur nos suites ; régime rare ≥ 15 suites
#   de fit. Seuils décidés avant les résultats ; le fit n'a produit aucun résultat de steering.
#
# Usage : python experiments/dose_reponse.py fit  --out-dir dr_0p6b --model Qwen/Qwen3-0.6B-Base --dtype bfloat16
#         python experiments/dose_reponse.py eval --out-dir dr_0p6b
#         python experiments/dose_reponse.py report --out-dir dr_0p6b

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from atlas.compat import capture_hooks_lot, construire_moteur, steer_hook  # noqa: E402
from reproduire_b2 import B2_MULTIMODAUX, PROMPTS, SEED  # noqa: E402
from trajectory_bifurcation_core import (  # noqa: E402
    acp,
    analyser_regimes,
    etiquettes_regimes,
    standardiser,
    test_bimodal,
)

NPC = 10
SEED_FIT = SEED + 17
SEED_EVAL = SEED + 99
EVAL_OFFSET = 10_000
ALPHAS_VRAI = (0.5, 1.0, 2.0, -1.0)
ALPHAS_CTRL = (1.0, 2.0)
CONTROLES = ("aleatoire", "permute", "autre")
SEUIL_FLUIDITE = 0.5          # nat par jeton
SEUIL_D = 0.10
SEUIL_P = 0.05
MIN_VALIDES = 6
SEUIL_DOSE = 0.70
SEUIL_FLUIDES = 0.50
SEUIL_CTRL = 0.05


# ---------------------------------------------------------------------------------------------------------------------
# Rejeu non perturbé : états moyens (R_h) et NLL par jeton
# ---------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def rejouer(eng, contexte: Sequence[int], suites: np.ndarray, couche: int, bs: Optional[int] = None):
    """Passe complète sans cache sur contexte + suite, modèle non perturbé. Renvoie, par suite : R_h (moyenne, sur les
    jetons générés, de la sortie de `couche` ; même représentation que rejouer_moyennes du moteur), NLL moyenne par jeton
    (loi du modèle à T = 1) et norme moyenne de l'état. Tous les jetons d'un lot partagent le même contexte (pas de
    remplissage)."""
    bs = bs or eng.batch_size
    T = suites.shape[1]
    R, NLL, NORM = [], [], []
    for b0 in range(0, len(suites), bs):
        s = suites[b0:b0 + bs]
        ids = torch.tensor([list(contexte) + x.tolist() for x in s], device=eng.device)
        with capture_hooks_lot(eng, [couche], last=T) as cap:
            z = eng.model(input_ids=ids, use_cache=False, logits_to_keep=T + 1).logits[:, :T, :].float()
        h = cap[couche]
        tok = torch.as_tensor(s, device=z.device)[..., None]
        lp = z.gather(-1, tok)[..., 0] - torch.logsumexp(z, dim=-1)
        R.append(h.mean(1).cpu().numpy())
        NORM.append(h.norm(dim=-1).mean(1).cpu().numpy())
        NLL.append((-lp).mean(1).cpu().numpy())
    return np.concatenate(R), np.concatenate(NLL), np.concatenate(NORM)


# ---------------------------------------------------------------------------------------------------------------------
# Classifieur des régimes, figé sur les suites de fit
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Classifieur:
    ref: np.ndarray
    V: np.ndarray
    centres: np.ndarray
    rare: int

    @classmethod
    def ajuster(cls, R: np.ndarray, seed: int) -> Tuple["Classifieur", np.ndarray]:
        """Étiquettes du moteur (2-moyennes sur l'ACP standardisée, comme B2) et classifieur à centres figés."""
        R = np.asarray(R, np.float64)
        lab = etiquettes_regimes(R, seed=seed, npc=NPC)
        Z = standardiser(R, R)
        V = acp(Z, NPC)
        P = Z @ V
        centres = np.stack([P[lab == k].mean(0) for k in (0, 1)])
        rare = int(np.argmin([(lab == 0).sum(), (lab == 1).sum()]))
        return cls(R, V, centres, rare), lab

    def etiqueter(self, X: np.ndarray) -> np.ndarray:
        """1 si la suite est du régime rare (centre le plus proche), sinon 0."""
        P = standardiser(np.asarray(X, np.float64), self.ref) @ self.V
        d = ((P[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1)
        return (d.argmin(1) == self.rare).astype(int)


def direction_vraie(R: np.ndarray, lab: np.ndarray, rare: int) -> np.ndarray:
    R = np.asarray(R, np.float64)
    return R[lab == rare].mean(0) - R[lab != rare].mean(0)


def mettre_a_la_norme(v: np.ndarray, norme: float) -> np.ndarray:
    return v * (norme / (np.linalg.norm(v) + 1e-12))


def controles(R: np.ndarray, lab: np.ndarray, rare: int, d_autre: Optional[np.ndarray], graine: int) -> Dict[str, np.ndarray]:
    """Directions de contrôle de même norme que la direction vraie : aléatoire, étiquettes permutées, autre prompt."""
    rng = np.random.default_rng(graine)
    norme = float(np.linalg.norm(direction_vraie(R, lab, rare)))
    out = {"aleatoire": mettre_a_la_norme(rng.standard_normal(R.shape[1]), norme)}
    perm = rng.permutation(lab)
    out["permute"] = mettre_a_la_norme(direction_vraie(R, perm, rare), norme)
    if d_autre is not None:
        out["autre"] = mettre_a_la_norme(d_autre, norme)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Phase fit
# ---------------------------------------------------------------------------------------------------------------------

def prompts_demandes(args) -> List[int]:
    return list(range(len(PROMPTS))) if not args.prompts else [int(i) for i in args.prompts]


def phase_fit(eng, args, couche: int):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = prompts_demandes(args)
    t0 = time.time()
    ens = eng.generer_ensemble([PROMPTS[i] for i in idx], args.n_fit, args.n_steps, seed_base=SEED_FIT, progress=True)
    arrays, rep, meta = {}, [], []
    for k, p in enumerate(idx):
        L = ens.lignes_du_prompt(k)
        tokens = ens.tokens[L]
        R, nll, norm = rejouer(eng, ens.prompt_ids[k], tokens, couche)
        arrays[f"R{p}"], arrays[f"tok{p}"], arrays[f"nll{p}"] = R.astype(np.float32), tokens, nll
        rep.append(R)
        dip, pv = test_bimodal(np.asarray(R, np.float64), SEED + k, NPC)
        meta.append({"idx": p, "dip": dip, "p": pv, "norme_etat": float(norm.mean()), "nll_ref": float(nll.mean())})
    cplet = len(idx) == len(PROMPTS)
    h = analyser_regimes(rep, seed=SEED, npc=NPC) if len(idx) >= 2 else None
    if h is not None:
        for m, flag in zip(meta, h["multimodal"]):
            m["multimodal_bh"] = bool(flag)
    info = {"modele": args.model, "dtype": args.dtype, "couche": couche, "n_fit": args.n_fit, "n_steps": args.n_steps,
            "seed": SEED, "seed_fit": SEED_FIT, "prompts_ajustes": idx, "ensemble_complet": cplet,
            "regimes": None if h is None else {k: h[k] for k in ("pi", "pi_ci", "n_multi", "nul", "puissance", "verdict")},
            "prompts": meta, "minutes": (time.time() - t0) / 60}
    np.savez_compressed(out / "fit.npz", **arrays)
    (out / "fit.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    n_multi = sum(m.get("multimodal_bh", False) for m in meta)
    print(f"[fit] {len(idx)} prompts, {n_multi} multimodaux (BH) ; régimes : "
          f"{None if h is None else h['verdict']} ; {info['minutes']:.1f} min", flush=True)
    return info


# ---------------------------------------------------------------------------------------------------------------------
# Phase eval
# ---------------------------------------------------------------------------------------------------------------------

def selectionner(info: dict, args) -> List[int]:
    flags = {m["idx"]: m for m in info["prompts"]}
    if args.selection == "forcee":
        return [int(i) for i in args.prompts]
    if args.selection == "b2":
        label = info["modele"].split("/")[-1]
        liste = B2_MULTIMODAUX.get(label)
        if liste is None:
            raise SystemExit(f"Pas de liste B2 publiée pour {label}")
        return [i for i in liste if i in flags and flags[i]["p"] < 0.05]
    if not info["ensemble_complet"]:
        raise SystemExit("--selection propre exige un fit sur les 64 prompts (BH sur l'ensemble)")
    return [i for i, m in flags.items() if m.get("multimodal_bh")]


def preparer(info: dict, donnees, idx: List[int], args) -> List[dict]:
    """Classifieur, direction vraie et contrôles de chaque prompt retenu (régime rare ≥ min_rare suites de fit)."""
    fits = []
    for p in idx:
        R = np.asarray(donnees[f"R{p}"], np.float64)
        cl, lab = Classifieur.ajuster(R, seed=SEED)
        n_rare = int((lab == cl.rare).sum())
        if n_rare < args.min_rare:
            print(f"[eval] prompt {p} exclu : régime rare de {n_rare} suites (< {args.min_rare})", flush=True)
            continue
        fits.append({"idx": p, "R": R, "lab": lab, "cl": cl, "n_rare": n_rare,
                     "d": direction_vraie(R, lab, cl.rare)})
    for k, f in enumerate(fits):
        autre = fits[(k + 1) % len(fits)]["d"] if len(fits) > 1 else None
        f["controles"] = controles(f["R"], f["lab"], f["cl"].rare, autre, SEED + 1000 + f["idx"])
    return fits


def evaluer_prompt(eng, f: dict, info: dict, args) -> dict:
    p, couche = f["idx"], info["couche"]
    cl, pid = f["cl"], list(eng.encoder(PROMPTS[p]))
    rows = [EVAL_OFFSET + p * 256 + j for j in range(args.n_eval)]

    def generer(vec, alpha, n=None):
        out = []
        for b0 in range(0, len(rows) if n is None else n, eng.batch_size):
            r = rows[b0:b0 + eng.batch_size][:(len(rows) if n is None else n) - b0]
            ctx = contextlib.nullcontext() if vec is None else steer_hook(eng, couche, vec, alpha=alpha,
                                                                         t_debut=args.t_debut)
            with ctx:
                out.append(eng.generer_lot([pid] * len(r), r, args.n_steps, seed_base=SEED_EVAL)["tokens"])
        return np.concatenate(out)

    ref = generer(None, 0.0)
    R0, nll0, _ = rejouer(eng, pid, ref, couche)
    y0 = cl.etiqueter(R0)
    r = {"idx": p, "prompt": PROMPTS[p], "n_rare_fit": f["n_rare"], "norme_d": float(np.linalg.norm(f["d"])),
         "taux_ref": float(y0.mean()), "nll_ref": float(nll0.mean()), "labels_ref": y0.tolist(), "conds": {}}
    ident = generer(f["d"], 0.0, n=min(eng.batch_size, len(rows)))
    r["identite_alpha0"] = bool((ident == ref[:len(ident)]).all())
    plan = [("vraie", f["d"], a) for a in ALPHAS_VRAI]
    plan += [(c, v, a) for c, v in f["controles"].items() for a in ALPHAS_CTRL]
    for nom, vec, alpha in plan:
        toks = generer(vec, alpha)
        R1, nll1, _ = rejouer(eng, pid, toks, couche)
        y1 = cl.etiqueter(R1)
        r["conds"][f"{nom}@{alpha:g}"] = {
            "direction": nom, "alpha": alpha, "taux": float(y1.mean()), "delta": float(y1.mean() - y0.mean()),
            "delta_nll": float(nll1.mean() - nll0.mean()), "frac_modifiees": float((toks != ref).any(1).mean()),
            "labels": y1.tolist()}
    return r


def phase_eval(eng, args, info: dict, donnees):
    out = Path(args.out_dir)
    chemin = out / "eval.json"
    idx = selectionner(info, args)
    fits = preparer(info, donnees, idx, args)
    res = {"fit": {k: info[k] for k in ("modele", "dtype", "couche", "n_fit", "n_steps")},
           "selection": args.selection, "n_eval": args.n_eval, "t_debut": args.t_debut,
           "prompts_selectionnes": [f["idx"] for f in fits], "prompts": {}}
    if chemin.exists():
        prev = json.loads(chemin.read_text())
        if prev.get("prompts_selectionnes") == res["prompts_selectionnes"] and prev.get("n_eval") == args.n_eval:
            res["prompts"] = prev["prompts"]
    for f in fits:
        if str(f["idx"]) in res["prompts"]:
            continue
        t0 = time.time()
        res["prompts"][str(f["idx"])] = evaluer_prompt(eng, f, info, args)
        chemin.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        v = res["prompts"][str(f["idx"])]["conds"]["vraie@1"]
        print(f"[eval] prompt {f['idx']:2d} ‖d‖ = {np.linalg.norm(f['d']):.1f} ; taux réf "
              f"{res['prompts'][str(f['idx'])]['taux_ref']:.2f} → vraie@1 {v['taux']:.2f} (ΔNLL {v['delta_nll']:+.2f}) ; "
              f"{time.time() - t0:.0f} s", flush=True)
    chemin.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


# ---------------------------------------------------------------------------------------------------------------------
# Statistiques et rapport
# ---------------------------------------------------------------------------------------------------------------------

def rangs(a: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, float)
    ordre = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[ordre[j + 1]] == a[ordre[i]]:
            j += 1
        r[ordre[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = rangs(x), rangs(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def wilcoxon_exact(d: Sequence[float]) -> float:
    """p unilatéral exact, H1 : d > 0 (zéros écartés, rangs moyens en cas d'égalité)."""
    d = np.array([x for x in d if x != 0.0], float)
    n = len(d)
    if n == 0:
        return 1.0
    r2 = np.round(2 * rangs(np.abs(d))).astype(int)               # rangs doublés : entiers
    obs = int(r2[d > 0].sum())
    tot = int(r2.sum())
    comptes = np.zeros(tot + 1)
    comptes[0] = 1.0
    for v in r2:
        comptes[v:] = comptes[v:] + comptes[:tot + 1 - v].copy()
    return float(comptes[obs:].sum() / comptes.sum())


def ic_apparie(y0: Sequence[int], y1: Sequence[int], B: int = 2000, seed: int = 0) -> Tuple[float, float]:
    y0, y1 = np.asarray(y0, float), np.asarray(y1, float)
    rng = np.random.default_rng(seed)
    n = len(y0)
    d = np.array([(y1[i] - y0[i]).mean() for i in (rng.integers(0, n, n) for _ in range(B))])
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def verdict(res: dict) -> dict:
    prompts = list(res["prompts"].values())
    valides, D, rho, ctrl = [], [], [], []
    for p in prompts:
        v1 = p["conds"]["vraie@1"]
        if v1["delta_nll"] > SEUIL_FLUIDITE:
            continue
        valides.append(p)
        cs = [p["conds"][f"{c}@1"]["delta"] for c in CONTROLES if f"{c}@1" in p["conds"]]
        ctrl.append(float(np.mean(cs)))
        D.append(v1["delta"] - ctrl[-1])
        alphas = [0.0, 0.5, 1.0, 2.0]
        taux = [p["taux_ref"]] + [p["conds"][f"vraie@{a:g}"]["taux"] for a in alphas[1:]]
        rho.append(spearman(alphas, taux))
    n, n_tot = len(valides), len(prompts)
    out = {"n_prompts": n_tot, "n_valides": n, "D": D, "rho": rho, "delta_controles": ctrl}
    if n < MIN_VALIDES:
        out.update(verdict="INDÉTERMINÉ", raison=f"{n} prompts valides (< {MIN_VALIDES})")
        return out
    med = float(np.median(D))
    pw = wilcoxon_exact(D)
    a = med >= SEUIL_D and pw < SEUIL_P
    b = float(np.mean([x > 0 for x in rho if not math.isnan(x)] or [0.0])) >= SEUIL_DOSE
    c = n / n_tot >= SEUIL_FLUIDES
    d = abs(float(np.median(ctrl))) < SEUIL_CTRL
    out.update({"P_A": a, "P_B": b, "P_C": c, "P_D": d, "mediane_D": med, "p_wilcoxon": pw,
                "part_dose_positive": float(np.mean([x > 0 for x in rho if not math.isnan(x)] or [0.0])),
                "part_fluides": n / n_tot, "mediane_controles": float(np.median(ctrl))})
    out["verdict"] = "CONFIRMÉ" if (a and b and c and d) else ("RÉFUTÉ" if not a else "PARTIEL")
    return out


def rapport(res: dict) -> str:
    v = verdict(res)
    L = [f"=== DOSE-RÉPONSE ({res['fit']['modele']}, {res['fit']['dtype']}, couche {res['fit']['couche']}, "
         f"sélection {res['selection']}, n_eval = {res['n_eval']}, t_debut = {res['t_debut']}) ==="]
    if res["selection"] == "forcee":
        L.append("ATTENTION : prompts forcés, hors critères de sélection (débogage) ; le verdict n'est pas valable.")
    L.append(f"Prompts évalués : {v['n_prompts']} ; identité à α = 0 : "
             f"{sum(p['identite_alpha0'] for p in res['prompts'].values())}/{v['n_prompts']}")
    L.append("")
    L.append(f"{'prompt':>6} {'réf':>5} " + " ".join(f"{'vraie@' + format(a, 'g'):>9}" for a in ALPHAS_VRAI)
             + f" {'ΔNLL@1':>7} " + " ".join(f"{c[:5] + '@1':>9}" for c in CONTROLES) + "   IC95 Δ vraie@1")
    for p in res["prompts"].values():
        c = p["conds"]
        lo, hi = ic_apparie(p["labels_ref"], c["vraie@1"]["labels"], seed=p["idx"])
        L.append(f"{p['idx']:>6} {p['taux_ref']:>5.2f} "
                 + " ".join(f"{c[f'vraie@{a:g}']['delta']:>+9.2f}" for a in ALPHAS_VRAI)
                 + f" {c['vraie@1']['delta_nll']:>+7.2f} "
                 + " ".join(f"{c[f'{k}@1']['delta']:>+9.2f}" if f"{k}@1" in c else f"{'—':>9}" for k in CONTROLES)
                 + f"   [{lo:+.2f}, {hi:+.2f}]")
    L.append("")
    L.append(f"Prompts valides (fluides à α = 1) : {v['n_valides']}/{v['n_prompts']}")
    if "mediane_D" in v:
        L.append(f"P-A efficacité : médiane D = {v['mediane_D']:+.3f} (≥ {SEUIL_D}), Wilcoxon p = {v['p_wilcoxon']:.4f} "
                 f"(< {SEUIL_P}) → {'oui' if v['P_A'] else 'non'}")
        L.append(f"P-B dose-réponse : ρ > 0 pour {v['part_dose_positive']:.0%} des prompts (≥ {SEUIL_DOSE:.0%}) → "
                 f"{'oui' if v['P_B'] else 'non'}")
        L.append(f"P-C fluidité : {v['part_fluides']:.0%} des prompts valides (≥ {SEUIL_FLUIDES:.0%}) → "
                 f"{'oui' if v['P_C'] else 'non'}")
        L.append(f"P-D spécificité : médiane des Δ de contrôle = {v['mediane_controles']:+.3f} (|.| < {SEUIL_CTRL}) → "
                 f"{'oui' if v['P_D'] else 'non'}")
    else:
        L.append(v["raison"])
    L.append(f"VERDICT : {v['verdict']}")
    ex = [p for p in res["prompts"].values()]
    neg = [p["conds"]["vraie@-1"]["delta"] for p in ex]
    L.append(f"Exploratoire : Δ(vraie, α = −1) < 0 pour {sum(x < 0 for x in neg)}/{len(neg)} prompts ; "
             f"suites modifiées à α = 1 : médiane {np.median([p['conds']['vraie@1']['frac_modifiees'] for p in ex]):.2f}")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------------

def charger(out_dir: str):
    info = json.loads((Path(out_dir) / "fit.json").read_text())
    return info, np.load(Path(out_dir) / "fit.npz")


def construire(args, info: Optional[dict] = None):
    modele = info["modele"] if info else args.model
    dtype = info["dtype"] if info else args.dtype
    eng = construire_moteur("sgz", model_name=modele, torch_dtype=getattr(torch, dtype), max_new_tokens=args.n_steps,
                            seed=SEED)
    couche = info["couche"] if info else (args.layer if args.layer is not None else (2 * eng.num_layers) // 3 - 1)
    return eng, couche


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fit", "eval", "report"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--layer", type=int, default=None, help="défaut : ⌊2L/3⌋ − 1 (hidden_states[⌊2L/3⌋] de B2)")
    ap.add_argument("--n-fit", type=int, default=192)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--t-debut", type=int, default=0)
    ap.add_argument("--prompts", nargs="*", default=None, help="indices de PROMPTS (fit restreint, ou --selection forcee)")
    ap.add_argument("--selection", default="propre", choices=["propre", "b2", "forcee"])
    ap.add_argument("--min-rare", type=int, default=15)
    args = ap.parse_args()

    if args.phase == "fit":
        eng, couche = construire(args)
        phase_fit(eng, args, couche)
    elif args.phase == "eval":
        info, donnees = charger(args.out_dir)
        args.n_steps = info["n_steps"]
        eng, _ = construire(args, info)
        res = phase_eval(eng, args, info, donnees)
        print(rapport(res))
    else:
        res = json.loads((Path(args.out_dir) / "eval.json").read_text())
        print(rapport(res))


if __name__ == "__main__":
    main()
