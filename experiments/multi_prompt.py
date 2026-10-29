# experiments/multi_prompt.py
# Expérience : 8 prompts × mode réflexion (Qwen3) activé / désactivé.
#
# Critères fixés avant l'exécution (unité statistique = le run prompt × mode) :
#   C1  Faiblesse 1 (la bifurcation = divergence des tokens) généralisée si
#       corr(JS(t), 1 - frac_même_préfixe) >= 0.9 dans >= 75 % des runs ;
#       réfutée si < 0.5 dans la majorité des runs.
#   C2  Faiblesse 2 corrigée si choose_patch_time trouve un t (>= 4 paires
#       informatives) dans >= 75 % des runs où le patching est tenté.
#   C3  Spécificité : témoins nuls (vecteur aléatoire de même norme ; état de
#       la source 8 pas plus tôt) avec |recovery_multistep| médian < 0.1.
#   C4  Faiblesse 3 (recovery binaire) atténuée si >= 25 % des paires
#       informatives ont recovery_multistep dans [0.1, 0.9].
#   Contrôles mécaniques : témoins positifs recovery_per_step[0] ~ 1 ;
#   témoins négatifs overlap = 1.
#
# Usage : python experiments/multi_prompt.py --out resultats.json

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_bifurcation_core import TrajectoryBifurcationEngine  # noqa: E402

PROMPTS = [
    "Raconte une histoire courte.",
    "Quelle est la capitale de l'Australie ?",
    "Donne-moi trois idées de cadeau pour un ami qui aime la cuisine.",
    "Explique pourquoi le ciel est bleu.",
    "Écris un poème de quatre vers sur l'hiver.",
    "Combien font 17 fois 23 ?",
    "Quel est le meilleur langage de programmation pour débuter ?",
    "Résume l'histoire du Petit Chaperon rouge.",
]


def run_one(engine, prompt, thinking, args, rng):
    kwargs = {"enable_thinking": thinking}
    trajs = engine.generate_ensemble(
        prompt, args.n, chat_template_kwargs=kwargs, seed=args.seed
    )
    div = engine.compute_divergences(trajs)
    js, frac = div["js_over_time"], div["frac_pairs_same_prefix"]
    corr = (
        float(np.corrcoef(js, 1 - frac)[0, 1])
        if np.std(js) > 0 and np.std(frac) > 0
        else None
    )
    bif = engine.detect_bifurcation(div, measure="js")
    t_star = bif["t_star"]
    t_patch = engine.choose_patch_time(
        trajs, t_min=t_star if t_star is not None else 1, n_pairs=args.n_pairs
    )
    out = {
        "prompt": prompt,
        "thinking": thinking,
        "min_len": div["min_len"],
        "corr_js_divergence_tokens": corr,
        "t_star": t_star,
        "baseline_degenerate": bif.get("baseline_degenerate"),
        "t_patch": t_patch,
        "layers": {},
        "nulls": [],
    }
    if t_patch is None:
        return out

    for layer in args.layers:
        results = engine.reversibility_analysis(
            trajs,
            t_patch=t_patch,
            layer=layer,
            n_pairs=args.n_pairs,
            n_controls=1,
            max_continuation_tokens=args.cont,
            forced_horizon=args.horizon,
        )
        out["layers"][layer] = [
            {
                "kind": r.pair_kind,
                "src": r.source_traj_idx,
                "tgt": r.target_traj_idx,
                "first_diff_step": r.first_diff_step,
                "last_token_same": r.last_token_same,
                "recovery": r.recovery,
                "recovery_step0": (r.recovery_per_step or [None])[0],
                "recovery_multistep": r.recovery_multistep,
                "recovery_loglik": r.recovery_loglik,
                "overlap": r.final_token_overlap,
            }
            for r in results
        ]

    # Témoins nuls à la couche du milieu, sur les paires informatives
    layer = args.layers[len(args.layers) // 2]
    for rec in out["layers"][layer]:
        if rec["kind"] != "informative":
            continue
        src, tgt = trajs[rec["src"]], trajs[rec["tgt"]]
        v = src.hidden_states[layer][t_patch]
        rand = rng.standard_normal(v.shape).astype(np.float32)
        rand *= np.linalg.norm(v) / np.linalg.norm(rand)
        nulls = {"aleatoire_meme_norme": rand}
        if t_patch >= 8:
            nulls["source_8_pas_avant"] = src.hidden_states[layer][t_patch - 8]
        for name, vec in nulls.items():
            eff = engine.forced_patch_effect(
                src, tgt, t_patch, layer, vector=vec, horizon=args.horizon
            )
            out["nulls"].append(
                {
                    "type": name,
                    "recovery_multistep": eff["recovery_multistep"],
                    "recovery_loglik": eff["recovery_loglik"],
                }
            )
    return out


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def report(runs, layers):
    print("\n=========== RÉSULTATS ===========")
    for thinking in (True, False):
        rs = [r for r in runs if r["thinking"] == thinking]
        label = "réflexion ON" if thinking else "réflexion OFF"
        corrs = [r["corr_js_divergence_tokens"] for r in rs]
        valid = [c for c in corrs if c is not None]
        print(f"\n--- {label} ({len(rs)} runs) ---")
        print(
            "C1 corr(JS, divergence tokens) par run :",
            [None if c is None else round(c, 3) for c in corrs],
        )
        print(
            f"   >= 0.9 : {sum(c >= 0.9 for c in valid)}/{len(valid)} ; "
            f"< 0.5 : {sum(c < 0.5 for c in valid)}/{len(valid)}"
        )
        print(
            "   t* :", [r["t_star"] for r in rs],
            "| t_patch :", [r["t_patch"] for r in rs],
        )
        found = sum(r["t_patch"] is not None for r in rs)
        print(f"C2 t_patch trouvé : {found}/{len(rs)}")
        for layer in layers:
            pairs = [
                p for r in rs for p in r["layers"].get(layer, [])
                if p["kind"] == "informative"
            ]
            per_run = [
                np.mean([p["recovery_multistep"] for p in r["layers"].get(layer, [])
                         if p["kind"] == "informative"
                         and p["recovery_multistep"] is not None] or [np.nan])
                for r in rs if r["layers"].get(layer)
            ]
            per_run = [x for x in per_run if not np.isnan(x)]
            ms = [p["recovery_multistep"] for p in pairs
                  if p["recovery_multistep"] is not None]
            mid = sum(0.1 <= x <= 0.9 for x in ms)
            print(
                f"   couche {layer:2d} : n_paires={len(pairs):2d} "
                f"recovery 1er pas méd={median([p['recovery'] for p in pairs])} "
                f"multi-pas méd={median(ms)} "
                f"loglik méd={median([p['recovery_loglik'] for p in pairs])} "
                f"| moyenne par run : min={min(per_run, default=None)} "
                f"max={max(per_run, default=None)} "
                f"| C4 dans [0.1,0.9] : {mid}/{len(ms)}"
            )
            # Exploratoire : dernier token identique (contexte seul) ou non
            for same in (True, False):
                sub = [p["recovery_multistep"] for p in pairs
                       if p["last_token_same"] is same
                       and p["recovery_multistep"] is not None]
                sub0 = [p["recovery"] for p in pairs
                        if p["last_token_same"] is same]
                print(f"      dernier token {'identique' if same else 'différent'} : "
                      f"n={len(sub)} 1er pas méd={median(sub0)} multi-pas méd={median(sub)}")
        for name in ("aleatoire_meme_norme", "source_8_pas_avant"):
            vals = [n["recovery_multistep"] for r in rs for n in r["nulls"]
                    if n["type"] == name]
            print(f"C3 témoin {name} : n={len(vals)} multi-pas méd={median(vals)} "
                  f"max |.|={max((abs(v) for v in vals if v is not None), default=None)}")
        pos = [p["recovery_step0"] for r in rs for L in r["layers"].values()
               for p in L if p["kind"] == "positive_control"]
        neg = [p["overlap"] for r in rs for L in r["layers"].values()
               for p in L if p["kind"] == "negative_control"]
        print(f"Contrôles : positifs pas 0 = {[round(x, 4) for x in pos if x is not None]} ; "
              f"négatifs overlap = {sorted(set(neg))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 7, 14, 21])
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--cont", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--modes", default="on,off")
    ap.add_argument("--out", default="resultats_multi_prompt.json")
    args = ap.parse_args()

    engine = TrajectoryBifurcationEngine(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        layers_to_track=sorted(set(args.layers)),
    )
    rng = np.random.default_rng(args.seed)
    modes = [m == "on" for m in args.modes.split(",")]
    runs = []
    for thinking in modes:
        for prompt in PROMPTS[: args.n_prompts]:
            t0 = time.time()
            runs.append(run_one(engine, prompt, thinking, args, rng))
            r = runs[-1]
            print(
                f"[{'ON ' if thinking else 'OFF'}] {prompt[:40]:40s} "
                f"t*={r['t_star']} t_patch={r['t_patch']} "
                f"corr={r['corr_js_divergence_tokens']} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            Path(args.out).write_text(json.dumps(runs, indent=1, default=str))
    report(runs, args.layers)


if __name__ == "__main__":
    main()
