# experiments/bassin_confirmation.py
# Test de confirmation du bassin d'Akarlar avec témoin « même classe », couche fixée d'avance et bootstrap des prompts.
# Critères complets, puissance et prédictions : experiments/BASSIN_CONFIRMATION.md (commit de ce fichier = date des
# critères). Montage de bassin_akarlar.py (cible reconstruite, patch persistant, nombres aléatoires communs), avec :
#   - patch « croisé » (source de l'autre classe) ET patch « même classe » (source de la même classe que la cible, avec un
#     premier jeton différent) aux mêmes couches ; le second mesure la perturbation due au patch lui-même ;
#   - statistique principale à la couche fixée (L24 de Qwen2.5-7B) :
#       E_C→H = logit p_H(croisé) − logit p_H(même classe)   sur les cibles correctes,
#       E_H→C = logit p_C(croisé) − logit p_C(même classe)   sur les cibles hallucinées,
#       A* = E_C→H − E_H→C ;
#   - IC95 par bootstrap à deux niveaux : prompts, puis paires dans chaque prompt (2000 tirages).
#
# Usage : python experiments/bassin_confirmation.py --model Qwen/Qwen2.5-7B --repartir --lot 96 --out conf.json

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import bassin_akarlar as ba  # noqa: E402
from trajectory_bifurcation_core import TrajectoryBifurcationEngine  # noqa: E402

SEED = 20261105
COUCHE = 24                    # couche principale, fixée d'avance
COUCHES = [20, 24, 27]         # 20 et 27 : descriptives
N_POOL, N_PAIRS, N_SEEDS = 60, 6, 24
B = 2000
MIN_UNITES = 40                # unités (paire, direction) avec source même classe, par direction
DIRS = (("C->H", "HALLUCINATION", "CORRECT"), ("H->C", "CORRECT", "HALLUCINATION"))   # direction, issue comptée, classe cible


class Confirmation(ba.Bassin):
    def __init__(self, eng, prompts, seed, textes=False):
        super().__init__(eng, prompts, n_pool=N_POOL, n_pairs=N_PAIRS, n_seeds=N_SEEDS, layers=COUCHES, l_soi=COUCHE)
        self.seed, self.textes = seed, textes

    def run(self):
        meta = []
        for pi, p in enumerate(self.prompts):
            t_start = time.time()
            pid = list(self.tok(p["prompt"])["input_ids"])
            pool = self.lots([pid] * self.n_pool, list(range(self.n_pool)), ba.MAX_GEN, self.seed + 100 * pi, 0)
            labs = [self.etiquettes(r, p)[0] for r in pool]
            t0 = [int(r[0]) for r in pool]
            prs = self.paires(labs, t0)
            m = {"idx": p["idx"], "pool": {c: labs.count(c) for c in ("CORRECT", "HALLUCINATION", "OTHER")},
                 "paires": len(prs)}
            meta.append(m)
            if not prs:
                print(f"[paires] {p['idx']} : aucune paire discordante", flush=True)
                continue
            st = self.etats([pid + [t] for t in t0])

            def meme_classe(tg):
                """Source même classe : premier run de la classe de tg, différent de tg, avec un autre premier jeton."""
                return next((k for k in range(len(labs)) if labs[k] == labs[tg] and k != tg and t0[k] != t0[tg]), None)

            jobs = []
            for j, (c, h) in enumerate(prs):
                for di, (d, _, _) in enumerate(DIRS):
                    tg, src = (c, h) if d == "C->H" else (h, c)
                    mc = meme_classe(tg)
                    for s in range(self.n_seeds):
                        jobs.append(dict(pair=j, dir=d, tg=tg, src=src, meme=mc, seed=s,
                                         row=((pi * 20 + j) * 2 + di) * 100 + s))
            m["unites_meme_classe"] = {d: sum(1 for x in jobs if x["dir"] == d and x["meme"] is not None) // self.n_seeds
                                       for d, _, _ in DIRS}
            conds = [("base", jobs, None, "tg"), ("echange", jobs, None, "src"), ("soi", jobs, COUCHE, "tg")]
            avec = [x for x in jobs if x["meme"] is not None]
            for l in COUCHES:
                conds += [(f"croise_L{l}", jobs, l, "src"), (f"meme_L{l}", avec, l, "meme")]
            for nom, js, layer, qui in conds:
                if not js:
                    continue
                rows = [x["row"] for x in js]
                if nom == "echange":
                    pref = [pid + [t0[x["src"]]] for x in js]
                else:
                    pref = [pid + [t0[x["tg"]]] for x in js]
                vecs = None if layer is None else st[layer][[x[qui] for x in js]]
                toks = self.lots(pref, rows, ba.MAX_GEN - 1, self.seed + 7919, 1, layer, vecs)
                for x, pr, tk in zip(js, pref, toks):
                    lab, strict, txt = self.etiquettes([pr[-1]] + [int(t) for t in tk], p)
                    self.rows.append(dict(idx=p["idx"], pair=x["pair"], dir=x["dir"], cond=nom, seed=x["seed"],
                                          label=lab, strict=strict, meme=x["meme"] is not None,
                                          **({"texte": txt} if self.textes else {})))
            print(f"[{p['idx']:2d}] réservoir {m['pool']} ; {len(prs)} paires ; même classe "
                  f"{m['unites_meme_classe']} ; {time.time() - t_start:.0f} s", flush=True)
        return meta


def logit(k, n):
    q = (k + 0.5) / (n + 1)
    return float(np.log(q / (1 - q)))


def analyser(rows, key="label", seed=SEED, b=B):
    """Statistiques par unité (prompt, paire, direction) ; bootstrap des prompts puis des paires."""
    par = {}
    for r in rows:
        cible = {d: o for d, o, _ in DIRS}[r["dir"]]
        k = (r["idx"], r["pair"], r["dir"], r["cond"])
        a, n = par.get(k, (0, 0))
        par[k] = (a + (r[key] == cible), n + 1)
    autre = {}
    for r in rows:
        k = (r["idx"], r["pair"], r["dir"], r["cond"])
        a, n = autre.get(k, (0, 0))
        autre[k] = (a + (r[key] == "OTHER"), n + 1)
    paires = sorted({(r["idx"], r["pair"]) for r in rows})
    avec = {(r["idx"], r["pair"], r["dir"]) for r in rows if r["meme"]}

    def stats(sel):
        def kn(d, c, sub=None, tab=par):
            v = [tab[(i, j, d, c)] for i, j in sel if (i, j, d, c) in tab and (sub is None or (i, j, d) in sub)]
            return sum(x for x, _ in v), sum(y for _, y in v)
        out = {}
        for d, _, _ in DIRS:
            o = {"E": {l: logit(*kn(d, f"croise_L{l}", avec)) - logit(*kn(d, f"meme_L{l}", avec)) for l in COUCHES},
                 "croise_vs_base": {l: logit(*kn(d, f"croise_L{l}")) - logit(*kn(d, "base")) for l in COUCHES},
                 "meme_vs_base": {l: logit(*kn(d, f"meme_L{l}", avec)) - logit(*kn(d, "base", avec)) for l in COUCHES},
                 "echange": logit(*kn(d, "echange")) - logit(*kn(d, "base"))}
            o["autre"] = {c: kn(d, c, None, autre)[0] / max(1, kn(d, c, None, autre)[1])
                          for c in ["base", "echange"] + [f"{t}_L{l}" for t in ("croise", "meme") for l in COUCHES]}
            out[d] = o
        out["A_etoile"] = {l: out["C->H"]["E"][l] - out["H->C"]["E"][l] for l in COUCHES}
        out["A_sans_temoin"] = {l: out["C->H"]["croise_vs_base"][l] - out["H->C"]["croise_vs_base"][l] for l in COUCHES}
        return out

    point = stats(paires)
    rng = np.random.default_rng(seed)
    strates = {}
    for i, j in paires:
        strates.setdefault(i, []).append((i, j))
    pl = list(strates)
    boots = []
    for _ in range(b):
        sel = []
        for q in rng.choice(len(pl), len(pl)):
            s = strates[pl[q]]
            sel += [s[k] for k in rng.integers(0, len(s), len(s))]
        boots.append(stats(sel))
    ci = lambda f: tuple(float(x) for x in np.percentile([f(x) for x in boots], [2.5, 97.5]))
    c = {"A_etoile": {l: ci(lambda x, l=l: x["A_etoile"][l]) for l in COUCHES},
         "A_sans_temoin": {l: ci(lambda x, l=l: x["A_sans_temoin"][l]) for l in COUCHES},
         **{f"{d}_E": {l: ci(lambda x, d=d, l=l: x[d]["E"][l]) for l in COUCHES} for d, _, _ in DIRS},
         **{f"{d}_echange": ci(lambda x, d=d: x[d]["echange"]) for d, _, _ in DIRS}}
    base = {(r["idx"], r["pair"], r["dir"], r["seed"]): r[key] for r in rows if r["cond"] == "base"}
    soi = [r for r in rows if r["cond"] == "soi"]
    soi_id = float(np.mean([base[(r["idx"], r["pair"], r["dir"], r["seed"])] == r[key] for r in soi])) if soi else None
    unites = {d: len({(i, j) for i, j, dd in avec if dd == d}) for d, _, _ in DIRS}
    return {"point": point, "ci": c, "soi_identique": soi_id, "unites_meme_classe": unites,
            "n_paires": len(paires), "n_prompts": len(pl)}


def verdict(res, min_unites=MIN_UNITES):
    c = res["ci"]
    if min(res["unites_meme_classe"].values()) < min_unites:
        return "NON TESTABLE (trop peu d'unités avec source même classe)"
    if res["soi_identique"] is not None and res["soi_identique"] < 0.98:
        return "NON INTERPRÉTABLE (auto-patch différent de la base)"
    if max(c["C->H_echange"][0], c["H->C_echange"][0]) <= 0:
        return "NON INTERPRÉTABLE (l'échange du premier jeton ne déplace pas l'issue)"
    a = c["A_etoile"][COUCHE]
    if a[0] > 0:
        return "ASYMÉTRIE COMME L'ARTICLE"
    if a[1] < 0:
        return "ASYMÉTRIE INVERSE"
    if -0.5 <= a[0] and a[1] <= 0.5:
        return "SYMÉTRIQUE"
    return "INDÉTERMINÉ"


def rapport(res, v):
    pt, c = res["point"], res["ci"]
    f = lambda x: f"{x:+.2f}"
    iv = lambda t: f"[{f(t[0])} ; {f(t[1])}]"
    L = [f"[PAIRES] {res['n_paires']} paires sur {res['n_prompts']} prompts ; unités avec source même classe : "
         f"{res['unites_meme_classe']}",
         f"[TÉMOIN −] auto-patch à L{COUCHE} identique à la base : {res['soi_identique']:.1%}"]
    for d, _, _ in DIRS:
        L.append(f"[{d}] échange de jeton Δ {f(pt[d]['echange'])} {iv(c[d + '_echange'])}")
        L.append(f"[{d}] E (croisé − même classe) : " + " ; ".join(
            f"L{l} {f(pt[d]['E'][l])} {iv(c[d + '_E'][l])}" for l in COUCHES))
        L.append(f"[{d}] croisé − base : " + " ; ".join(f"L{l} {f(pt[d]['croise_vs_base'][l])}" for l in COUCHES)
                 + " | même classe − base : " + " ; ".join(f"L{l} {f(pt[d]['meme_vs_base'][l])}" for l in COUCHES))
        L.append(f"[{d}] part « Autre » : " + " ; ".join(f"{k} {x:.2f}" for k, x in pt[d]["autre"].items()))
    L.append("[A*] asymétrie corrigée par le témoin même classe : " + " ; ".join(
        f"L{l} {f(pt['A_etoile'][l])} {iv(c['A_etoile'][l])}" for l in COUCHES))
    L.append("[A sans témoin, bootstrap des prompts] " + " ; ".join(
        f"L{l} {f(pt['A_sans_temoin'][l])} {iv(c['A_sans_temoin'][l])}" for l in COUCHES))
    L.append(f"[VERDICT] couche fixée L{COUCHE} : {v}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--prompts", default=str(ROOT / "experiments" / "data" / "akarlar_all_prompts.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--repartir", action="store_true")
    ap.add_argument("--lot", type=int, default=None)
    ap.add_argument("--out", default="bassin_confirmation.json")
    ap.add_argument("--textes", action="store_true", help="garder le texte de chaque suite (validation du classifieur)")
    ap.add_argument("--couches", type=int, nargs="+", default=None,
                    help="sous-ensemble des couches (doit contenir la couche fixée) ; les autres conditions sont inchangées")
    args = ap.parse_args()
    global COUCHES
    if args.couches:
        if COUCHE not in args.couches:
            raise SystemExit(f"--couches doit contenir L{COUCHE}")
        COUCHES = sorted(args.couches)
    modele = tok = None
    if args.repartir:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = torch.float16 if torch.cuda.is_available() else torch.float32
        modele = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, device_map="auto").eval()
        tok = AutoTokenizer.from_pretrained(args.model)
    eng = TrajectoryBifurcationEngine(model_name=args.model, sampler="gumbel", temperature=ba.TEMPERATURE, top_p=1.0,
                                      chat=False, forbid_special=False, seed=args.seed, max_new_tokens=ba.MAX_GEN,
                                      model=modele, tokenizer=tok, batch_size=args.lot)
    if COUCHE >= eng.num_layers:
        raise SystemExit(f"le modèle n'a que {eng.num_layers} couches ; la couche fixée L{COUCHE} n'existe pas")
    t0 = time.time()
    prompts = json.loads(Path(args.prompts).read_text())
    b = Confirmation(eng, prompts, args.seed, textes=args.textes)
    meta = b.run()
    out = {"setup": {"model": args.model, "seed": args.seed, "couche": COUCHE, "couches": COUCHES, "n_pool": N_POOL,
                     "n_pairs": N_PAIRS, "n_seeds": N_SEEDS, "lot": eng.batch_size, "dtype": str(eng.model.dtype),
                     "torch": torch.__version__},
           "prompts": meta, "lignes": b.rows,
           "textes_prompts": {p["idx"]: p["prompt"] for p in prompts} if args.textes else None}
    if not b.rows:
        out["verdict"] = "NON TESTABLE (aucune paire)"
        print(f"[VERDICT] {out['verdict']}")
    else:
        res = analyser(b.rows)
        v = verdict(res)
        res_s = analyser(b.rows, key="strict")
        out.update({"resultats": res, "verdict": v, "resultats_strict": res_s, "verdict_strict": verdict(res_s)})
        print(rapport(res, v))
        print(f"[STRICT] A* L{COUCHE} = {res_s['point']['A_etoile'][COUCHE]:+.2f} "
              f"[{res_s['ci']['A_etoile'][COUCHE][0]:+.2f} ; {res_s['ci']['A_etoile'][COUCHE][1]:+.2f}] → {out['verdict_strict']}")
    out["setup"]["minutes"] = (time.time() - t0) / 60
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, default=float))
    print(f"[SORTIE] {args.out} ({out['setup']['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
