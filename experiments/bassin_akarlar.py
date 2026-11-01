# experiments/bassin_akarlar.py
# Test propre de l'hypothèse du « bassin de l'hallucination » d'Akarlar (2026, arXiv 2604.15400), sur son montage :
# Qwen2.5-1.5B (base), ses 8 prompts « les plus équilibrés », son classifieur par sous-chaînes, T = 0,7, 15 jetons au plus,
# arrêt sur fin de texte ou deux retours à la ligne. Ce que le code publié ne fait pas et que ce test fait :
#   - la cible est une VRAIE trajectoire (son propre premier jeton t0), pas un nouveau tirage ;
#   - le patch PERSISTE : l'état de la couche l à la position de t0 est remplacé par celui de la source au préremplissage,
#     et les clés/valeurs des couches suivantes restent dans le cache pour toute la suite ;
#   - nombres aléatoires communs : pour une paire, une direction et une graine, toutes les conditions (sans patch, patch à
#     chaque couche, échange de jeton) reçoivent le même bruit de Gumbel à chaque pas ;
#   - couches fixées d'avance, dont les deux « couches d'engagement » de l'article (L20, L24) ;
#   - comparaison en log-odds, chaque direction rapportée à SA référence sans patch.
#
# CRITÈRES FIXÉS AVANT EXÉCUTION (commit de ce fichier)
#   Paires : par prompt, jusqu'à 6 paires (run correct, run halluciné) de premiers jetons différents, tirées d'un réservoir
#     de 60 trajectoires naturelles (Gumbel, graines fixes). Unité statistique : la paire.
#   Directions : C→H (cible correcte, source hallucinée ; issue comptée = HALLUCINATION) et H→C (cible hallucinée, source
#     correcte ; issue comptée = CORRECT). 24 graines de suite par paire et par direction.
#   Conditions : base (cible sans patch), patch persistant aux couches 0, 6, 12, 18, 20, 24, 27, échange de jeton (t0 de
#     la cible remplacé par celui de la source), auto-patch à L20 (état de la cible elle-même ; témoin négatif).
#   Δ(cond) = logit p(cond) − logit p(base), p = taux de l'issue comptée sur toutes les paires ((k + 0,5) / (n + 1)).
#   Statistique principale : A = moyenne sur les 7 couches de Δ_C→H − moyenne de Δ_H→C. IC95 : bootstrap des paires,
#     stratifié par prompt (2000 tirages).
#   Témoins :
#     T- auto-patch : étiquettes identiques à la base dans ≥ 98 % des suites, sinon NON INTERPRÉTABLE (montage défectueux).
#     T+ échange de jeton : borne basse de l'IC de Δ_échange > 0 dans au moins une direction, sinon NON INTERPRÉTABLE (le
#        premier jeton n'engage pas l'issue : aucun patch au pas 1 ne peut le faire).
#   Verdict sur l'asymétrie (l'article prédit A > 0 : corrompre plus facile que corriger) :
#     ASYMÉTRIE COMME L'ARTICLE  si borne basse de A > 0 et borne basse de Δ_C→H moyen > 0 ;
#     ASYMÉTRIE INVERSE          si borne haute de A < 0 ;
#     SYMÉTRIQUE                 si l'IC de A tient dans [−0,5 ; +0,5] ;
#     INDÉTERMINÉ                sinon.
#   Secondaire (confirmation du signal P2 de la réplication) : borne basse de Δ_H→C moyen > 0 → P2 CONFIRMÉ ;
#     IC dans [−0,3 ; +0,3] → P2 NON CONFIRMÉ ; sinon INDÉTERMINÉ.
#   Descriptifs : Δ par couche et par direction, classement strict (mots entiers), part des suites où le deuxième jeton
#     est le même sans patch et après échange du premier jeton.
#
# Usage (Kaggle T4 ou CPU) : python experiments/bassin_akarlar.py --out bassin_akarlar.json
# Durée : ~10-20 min sur une T4, de l'ordre d'une heure ou deux sur CPU [estimation, non mesurée].

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from trajectory_bifurcation_core import TrajectoryBifurcationEngine  # noqa: E402
from replication_akarlar import classify, truncate  # noqa: E402

SEED = 20261101
LAYERS = [0, 6, 12, 18, 20, 24, 27]
L_SOI = 20
N_POOL, N_PAIRS, N_SEEDS, MAX_GEN, TEMPERATURE = 60, 6, 24, 15, 0.7
B = 2000
DIRECTIONS = (("C->H", "HALLUCINATION"), ("H->C", "CORRECT"))


@contextlib.contextmanager
def patch_premier_passage(eng, layer, vecs):
    """Remplace, au préremplissage seulement, la sortie de `layer` à la dernière position par vecs (une ligne par suite).
    Les couches suivantes calculent leurs clés/valeurs depuis l'état patché : le patch persiste dans le cache."""
    etat = {"fait": False}

    def hook(module, inputs, output):
        if etat["fait"]:
            return output
        h, rest = eng._split_output(output)
        h = h.clone()
        h[:, -1, :] = vecs.to(device=h.device, dtype=h.dtype)
        etat["fait"] = True
        return h if rest is None else (h,) + rest

    handle = eng._decoder_layers()[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class Bassin:
    def __init__(self, eng, prompts, n_pool=N_POOL, n_pairs=N_PAIRS, n_seeds=N_SEEDS, layers=LAYERS, l_soi=L_SOI):
        self.eng, self.tok, self.prompts = eng, eng.tokenizer, prompts
        self.n_pool, self.n_pairs, self.n_seeds = n_pool, n_pairs, n_seeds
        self.layers, self.l_soi = [l for l in layers if l < eng.num_layers], l_soi if l_soi < eng.num_layers else layers[0]
        self.rows = []

    def etiquettes(self, ids, p):
        txt = self.tok.decode(truncate(self.tok, list(ids)))
        return (classify(txt, p["ground_truth"], p["wrong_indicators"]),
                classify(txt, p["ground_truth"], p["wrong_indicators"], strict=True), txt)

    def lots(self, contexts, rows, n_steps, seed_base, step0, layer=None, vecs=None):
        """Génération Gumbel par lots ; patch persistant si layer est donné."""
        bs, out = self.eng.batch_size, []
        for b0 in range(0, len(contexts), bs):
            sl = slice(b0, b0 + bs)
            ctx = (patch_premier_passage(self.eng, layer, vecs[sl]) if layer is not None else contextlib.nullcontext())
            with ctx:
                out.append(self.eng.generer_lot(contexts[sl], rows[sl], n_steps, seed_base=seed_base, step0=step0)["tokens"])
        return np.concatenate(out)

    @torch.no_grad()
    def etats(self, prefixes):
        """Sortie de chaque couche de LAYERS (+ L_SOI) à la dernière position de chaque préfixe (mêmes longueurs)."""
        couches = sorted(set(self.layers) | {self.l_soi})
        res = {l: [] for l in couches}
        bs = self.eng.batch_size
        for b0 in range(0, len(prefixes), bs):
            ids = torch.tensor(prefixes[b0:b0 + bs], device=self.eng.device)
            with self.eng._capture_hooks_batch(couches) as cap:
                self.eng.model(input_ids=ids, use_cache=False)
            for l in couches:
                res[l].append(cap[l][:, -1, :].cpu())
        return {l: torch.cat(v) for l, v in res.items()}

    def paires(self, labels, t0):
        C = [i for i, l in enumerate(labels) if l == "CORRECT"]
        H = [i for i, l in enumerate(labels) if l == "HALLUCINATION"]
        cand = [(c, h) for c in C for h in H if t0[c] != t0[h]]
        out, uc, uh = [], set(), set()
        for c, h in cand:                                   # d'abord sans réutiliser un run
            if c not in uc and h not in uh and len(out) < self.n_pairs:
                out.append((c, h)); uc.add(c); uh.add(h)
        for c, h in cand:                                   # puis en complétant
            if len(out) >= self.n_pairs:
                break
            if (c, h) not in out:
                out.append((c, h))
        return out

    def run(self):
        meta = []
        for pi, p in enumerate(self.prompts):
            t_start = time.time()
            pid = list(self.tok(p["prompt"])["input_ids"])
            pool = self.lots([pid] * self.n_pool, list(range(self.n_pool)), MAX_GEN, SEED + 100 * pi, 0)
            labs = [self.etiquettes(r, p)[0] for r in pool]
            t0 = [int(r[0]) for r in pool]
            prs = self.paires(labs, t0)
            meta.append({"idx": p["idx"], "pool": {c: labs.count(c) for c in ("CORRECT", "HALLUCINATION", "OTHER")},
                         "paires": len(prs)})
            if not prs:
                print(f"[paires] {p['idx']} : aucune paire discordante", flush=True)
                continue
            st = self.etats([pid + [t] for t in t0])
            jobs = []                                       # (paire, direction, graine) → cible, source
            for j, (c, h) in enumerate(prs):
                for di, (d, cible) in enumerate(DIRECTIONS):
                    tg, src = (c, h) if d == "C->H" else (h, c)
                    for s in range(self.n_seeds):
                        jobs.append(dict(pair=j, dir=d, cible=cible, tg=tg, src=src, seed=s,
                                         row=((pi * 20 + j) * 2 + di) * 100 + s))
            rows = [x["row"] for x in jobs]
            pref_tg = [pid + [t0[x["tg"]]] for x in jobs]
            pref_src = [pid + [t0[x["src"]]] for x in jobs]
            conds = [("base", pref_tg, None, None), ("echange", pref_src, None, None),
                     ("soi", pref_tg, self.l_soi, st[self.l_soi][[x["tg"] for x in jobs]])]
            conds += [(f"L{l}", pref_tg, l, st[l][[x["src"] for x in jobs]]) for l in self.layers]
            for nom, pref, layer, vecs in conds:
                toks = self.lots(pref, rows, MAX_GEN - 1, SEED + 7919, 1, layer, vecs)
                for x, pr, tk in zip(jobs, pref, toks):
                    lab, strict, _ = self.etiquettes([pr[-1]] + [int(t) for t in tk], p)
                    self.rows.append(dict(idx=p["idx"], pair=x["pair"], dir=x["dir"], cond=nom, seed=x["seed"],
                                          label=lab, strict=strict, t1=int(tk[0])))
            print(f"[{p['idx']:2d}] réservoir {meta[-1]['pool']} ; {len(prs)} paires ; {time.time() - t_start:.0f} s",
                  flush=True)
        return meta


# ---------------------------------------------------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------------------------------------------------

def logit(k, n):
    p = (k + 0.5) / (n + 1)
    return float(np.log(p / (1 - p)))


def analyser(rows, layers, l_soi, key="label"):
    cles = sorted({(r["idx"], r["pair"]) for r in rows})
    par = {}                                                # (idx, pair, dir, cond) → (k, n)
    for r in rows:
        cible = dict(DIRECTIONS)[r["dir"]]
        k, n = par.get((r["idx"], r["pair"], r["dir"], r["cond"]), (0, 0))
        par[(r["idx"], r["pair"], r["dir"], r["cond"])] = (k + (r[key] == cible), n + 1)

    def stats(sel):
        def kn(d, c):
            v = [par[(i, j, d, c)] for i, j in sel if (i, j, d, c) in par]
            return sum(a for a, _ in v), sum(b for _, b in v)
        out = {}
        for d, _ in DIRECTIONS:
            kb, nb = kn(d, "base")
            out[d] = {c: logit(*kn(d, c)) - logit(kb, nb) for c in ["echange"] + [f"L{l}" for l in layers]}
            out[d]["moy"] = float(np.mean([out[d][f"L{l}"] for l in layers]))
            out[d]["p"] = {c: kn(d, c)[0] / max(kn(d, c)[1], 1) for c in ["base", "echange"] + [f"L{l}" for l in layers]}
        out["A"] = out["C->H"]["moy"] - out["H->C"]["moy"]
        return out

    point = stats(cles)
    rng = np.random.default_rng(SEED)
    strates = {}
    for i, j in cles:
        strates.setdefault(i, []).append((i, j))
    boots = []
    for _ in range(B):
        sel = [s[k] for s in strates.values() for k in rng.integers(0, len(s), len(s))]
        boots.append(stats(sel))
    ci = lambda f: tuple(float(x) for x in np.percentile([f(b) for b in boots], [2.5, 97.5]))
    res = {"point": point,
           "ci": {"A": ci(lambda b: b["A"]),
                  **{f"{d}_moy": ci(lambda b, d=d: b[d]["moy"]) for d, _ in DIRECTIONS},
                  **{f"{d}_echange": ci(lambda b, d=d: b[d]["echange"]) for d, _ in DIRECTIONS},
                  **{f"{d}_L{l}": ci(lambda b, d=d, l=l: b[d][f"L{l}"]) for d, _ in DIRECTIONS for l in layers}}}
    base = {(r["idx"], r["pair"], r["dir"], r["seed"]): r[key] for r in rows if r["cond"] == "base"}
    soi = [r for r in rows if r["cond"] == "soi"]
    res["soi_identique"] = float(np.mean([base[(r["idx"], r["pair"], r["dir"], r["seed"])] == r[key] for r in soi])) \
        if soi else None
    c = res["ci"]
    if res["soi_identique"] is not None and res["soi_identique"] < 0.98:
        v = "NON INTERPRÉTABLE (auto-patch différent de la base : montage défectueux)"
    elif max(c["C->H_echange"][0], c["H->C_echange"][0]) <= 0:
        v = "NON INTERPRÉTABLE (l'échange du premier jeton ne déplace pas l'issue)"
    elif c["A"][0] > 0 and c["C->H_moy"][0] > 0:
        v = "ASYMÉTRIE COMME L'ARTICLE"
    elif c["A"][1] < 0:
        v = "ASYMÉTRIE INVERSE"
    elif -0.5 <= c["A"][0] and c["A"][1] <= 0.5:
        v = "SYMÉTRIQUE"
    else:
        v = "INDÉTERMINÉ"
    hc = c["H->C_moy"]
    p2 = "P2 CONFIRMÉ" if hc[0] > 0 else ("P2 NON CONFIRMÉ" if -0.3 <= hc[0] and hc[1] <= 0.3 else "INDÉTERMINÉ")
    res["verdict"], res["verdict_p2"] = v, p2
    return res


def rapport(res, layers):
    pt, c = res["point"], res["ci"]
    f = lambda x: f"{x:+.2f}"
    lignes = [f"[TÉMOIN −] auto-patch à L{L_SOI} identique à la base : {res['soi_identique']:.1%}"]
    for d, _ in DIRECTIONS:
        lignes.append(f"[{d}] taux sans patch {pt[d]['p']['base']:.2f} ; échange de jeton {pt[d]['p']['echange']:.2f} "
                      f"(Δ {f(pt[d]['echange'])} [{f(c[d + '_echange'][0])} ; {f(c[d + '_echange'][1])}])")
        lignes.append(f"[{d}] Δ log-odds par couche : " + " ; ".join(
            f"L{l} {f(pt[d][f'L{l}'])} [{f(c[f'{d}_L{l}'][0])} ; {f(c[f'{d}_L{l}'][1])}]" for l in layers))
        lignes.append(f"[{d}] Δ moyen {f(pt[d]['moy'])} [{f(c[d + '_moy'][0])} ; {f(c[d + '_moy'][1])}]")
    lignes.append(f"[A] asymétrie Δ_C→H − Δ_H→C = {f(pt['A'])} [{f(c['A'][0])} ; {f(c['A'][1])}] → {res['verdict']}")
    lignes.append(f"[P2] Δ_H→C moyen → {res['verdict_p2']}")
    return "\n".join(lignes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompts", default=str(ROOT / "experiments" / "data" / "akarlar_selected_prompts.json"))
    ap.add_argument("--dtype", default=None, choices=[None, "float16", "bfloat16", "float32"])
    ap.add_argument("--out", default="bassin_akarlar.json")
    args = ap.parse_args()
    dt = getattr(torch, args.dtype) if args.dtype else None    # défaut du moteur : fp16 sur T4, fp32 sur CPU
    eng = TrajectoryBifurcationEngine(model_name=args.model, sampler="gumbel", temperature=TEMPERATURE, top_p=1.0,
                                      chat=False, forbid_special=False, torch_dtype=dt, seed=SEED,
                                      max_new_tokens=MAX_GEN)
    t0 = time.time()
    b = Bassin(eng, json.loads(Path(args.prompts).read_text()))
    meta = b.run()
    res = analyser(b.rows, b.layers, b.l_soi)
    res_strict = analyser(b.rows, b.layers, b.l_soi, key="strict")
    print(rapport(res, b.layers))
    print(f"[STRICT] classement en mots entiers : A = {res_strict['point']['A']:+.2f} → {res_strict['verdict']}")
    t1 = {}
    for r in b.rows:
        if r["cond"] in ("base", "echange"):
            t1.setdefault((r["idx"], r["pair"], r["dir"], r["seed"]), {})[r["cond"]] = r["t1"]
    meme_t1 = float(np.mean([v["base"] == v["echange"] for v in t1.values() if len(v) == 2])) if t1 else None
    print(f"[DESCRIPTIF] même deuxième jeton sans patch et après échange du premier : {meme_t1:.1%}")
    Path(args.out).write_text(json.dumps({
        "setup": {"model": args.model, "dtype": str(eng.model.dtype), "layers": b.layers, "seed": SEED,
                  "n_pool": N_POOL, "n_pairs": N_PAIRS, "n_seeds": N_SEEDS, "torch": torch.__version__,
                  "minutes": (time.time() - t0) / 60},
        "prompts": meta, "resultats": res, "resultats_strict": res_strict, "lignes": b.rows,
        "n_suites": len(b.rows), "meme_deuxieme_jeton": meme_t1}, ensure_ascii=False, default=float))
    print(f"[SORTIE] {args.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
