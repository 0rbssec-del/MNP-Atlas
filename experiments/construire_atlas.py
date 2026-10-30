# experiments/construire_atlas.py
# Première carte de l'espace caché (atlas v0.4) : les 64 prompts de B2, 192 suites de 32 jetons à T = 0,6 (protocole
# sgz), même bruit que B2 (base SEED + 17, identifiant p·256 + j) et même représentation R_h (passe rejouée, couche
# ⌊2L/3⌋ − 1 du moteur), même test (analyser_regimes, graine SEED). La carte ajoute à B2, pour chaque prompt multimodal,
# la bifurcation par préfixe et la valeur ajoutée de l'état au pas 8 sur un lecteur de jetons.
# Contrôle : décisions « multimodal » comparées, prompt par prompt, à celles publiées par B2 (64/64 attendu avec les
# mêmes versions de torch et transformers) ; ce run remplace donc aussi reproduire_b2.py.
#
# Usage (GPU, Internet), deux modèles en même temps, un par GPU (Kaggle T4 × 2) :
#   python experiments/construire_atlas.py --model Qwen/Qwen3-1.7B-Base HuggingFaceTB/SmolLM3-3B-Base --out atlas_b2
# Sorties : atlas_b2/carte_<modèle>.json (une par modèle), puis atlas_b2/atlas/ (atlas v0.4 fusionné) et le résumé,
# dont l'accord entre les deux cartes (B2 après coup : 5 prompts partagés, p = 0,009).
# Durée : ~20 min (Qwen3-1.7B) et ~30 min (SmolLM3-3B) en parallèle sur une T4 chacun [estimation].

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from atlas.compat import construire_moteur  # noqa: E402
from atlas.bifurcation_atlas import BifurcationAtlas  # noqa: E402
from reproduire_b2 import B2_MULTIMODAUX, PROMPTS, SEED  # noqa: E402


def une_carte(model, n, steps, out_dir):
    label = model.split("/")[-1]
    eng = construire_moteur("sgz", model_name=model, max_new_tokens=steps, seed=SEED)
    if eng.tokenizer.pad_token_id is None:                  # comme B2 : remplissage par la fin de texte
        eng.tokenizer.pad_token = eng.tokenizer.eos_token
        eng.pad_id = eng.tokenizer.pad_token_id
    t0 = time.time()
    atlas = BifurcationAtlas(f"b2_{label}")
    i = atlas.cartographier(eng, PROMPTS, nom=label, n=n, n_steps=steps, seed_base=SEED + 17, seed_analyse=SEED,
                            rejouer=True)
    c = atlas.resume_carte(i)
    r = c["regimes"]
    print(f"[CARTE] {label} : {c['n_multimodaux']}/{c['n_prompts']} prompts multimodaux (π = {r['pi']:.2f}) ; substitut "
          f"gaussien {r['nul']:.0%} ; mélanges {r['puissance']:.0%} → {r['verdict']} ({(time.time() - t0) / 60:.1f} min)",
          flush=True)
    print(f"[CARTE] {label} valeur ajoutée des états sur les jetons (prompts multimodaux) : {c['valeur_ajoutee']} ; "
          f"préfixe décisif {c['prefixe_k_trouve']}/{c['n_multimodaux']}", flush=True)
    if label in B2_MULTIMODAUX and n == 192 and steps == 32:
        ref = np.zeros(len(PROMPTS), bool)
        ref[B2_MULTIMODAUX[label]] = True
        mm = np.array([f["multimodal"] for f in atlas.cartes[i]["carte"]["fiches"]])
        print(f"[VÉRIF] {label} décisions identiques à B2 : {int((mm == ref).sum())}/{len(PROMPTS)} prompts "
              f"(64/64 attendu avec les mêmes versions de torch et transformers)", flush=True)
    atlas.exporter_carte(i, out_dir / f"carte_{label}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"])
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", default="atlas_b2")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(args.model) == 1:
        une_carte(args.model[0], args.n, args.steps, out_dir)
        return

    # Un processus par modèle, chacun ne voyant qu'un GPU ; sans GPU, un modèle à la fois
    n_gpu = torch.cuda.device_count()
    lot = max(n_gpu, 1)
    procs, codes = [], []
    for i, m in enumerate(args.model):
        env = dict(os.environ)
        if n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(i % n_gpu)
        print(f"[PARALLÈLE] {m} → GPU {env.get('CUDA_VISIBLE_DEVICES', 'aucun')}", flush=True)
        procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--model", m, "--n", str(args.n),
                                       "--steps", str(args.steps), "--out", str(out_dir)], env=env))
        if len(procs) == lot:
            codes += [p.wait() for p in procs]
            procs = []
    codes += [p.wait() for p in procs]
    fusionner(args.model, out_dir)
    if any(codes):
        raise SystemExit(f"un des processus a échoué (codes {codes})")


def fusionner(models, out_dir):
    """Cartes des processus → un atlas v0.4, son résumé et l'accord entre cartes."""
    atlas = BifurcationAtlas("b2")
    for m in models:
        f = out_dir / f"carte_{m.split('/')[-1]}.json"
        if f.exists():
            atlas.importer_carte(f)
    atlas.save(out_dir / "atlas", overwrite=True)
    atlas.print_summary()
    for i in range(len(atlas.cartes)):
        for j in range(i + 1, len(atlas.cartes)):
            a = atlas.accord_cartes(i, j)
            print(f"[ACCORD] prompts multimodaux partagés : {a['prompts_partages']} ; comparabilité "
                  f"{a['comparabilite']}")
    (out_dir / "resume.json").write_text(json.dumps(atlas.summary(), indent=1, ensure_ascii=False, default=float))
    return atlas


if __name__ == "__main__":
    main()
