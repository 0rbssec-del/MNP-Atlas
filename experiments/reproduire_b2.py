# experiments/reproduire_b2.py
# Reproduit B2 (SGZ, « les trajectoires d'un même prompt forment-elles des bassins ? ») avec le moteur v0.2, à T = 0,6 :
# mêmes 64 prompts, 192 suites de 32 jetons, même bruit (base SEED + 17, identifiant p·256 + j), mêmes lots, même
# représentation R_h (moyenne des états après ⌊2L/3⌋ couches, passe rejouée), même test (analyser_regimes, graine SEED).
# Compare ensuite, prompt par prompt, la décision « multimodal » à celle publiée par B2.
#
# Usage (GPU, Internet) : python experiments/reproduire_b2.py --model Qwen/Qwen3-1.7B-Base --out b2_qwen.json
# Durée : ~15 min (Qwen3-1.7B) ou ~20 min (SmolLM3-3B) sur une T4 [estimation].

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_bifurcation_core import TrajectoryBifurcationEngine, analyser_regimes  # noqa: E402

SEED = 20261029
PROMPTS = [
    "La ville de Lyon est située au confluent", "Le principe de conservation de l'énergie affirme que",
    "Hier soir, en rentrant du travail, Marie a découvert", "Pour préparer une pâte à crêpes, il faut d'abord",
    "L'histoire de la Révolution française commence", "Les réseaux de neurones artificiels sont des modèles",
    "Le chat s'approcha lentement de la fenêtre et", "Dans un article publié la semaine dernière, les chercheurs",
    "La photosynthèse est le processus par lequel", "Il était une fois, dans un petit village de montagne,",
    "Le théorème de Pythagore permet de calculer", "Selon les prévisions météorologiques, demain il",
    "Le marché boursier a connu une forte baisse", "Pour installer ce logiciel, ouvrez un terminal et",
    "La musique baroque se caractérise par", "Le fleuve Saint-Laurent traverse", "Mon grand-père me racontait souvent que",
    "Les volcans se forment lorsque", "La recette traditionnelle du pot-au-feu", "En informatique, une fonction récursive est",
    "Le Québec compte plusieurs régions, dont", "La lumière se propage dans le vide à", "Le roman commence par une scène où",
    "Les abeilles jouent un rôle essentiel dans", "Le conseil municipal a voté hier une résolution",
    "Une étoile à neutrons est un objet", "Le premier jour d'école, Lucas avait peur de",
    "Pour résoudre une équation du second degré,", "La cathédrale Notre-Dame de Paris a été",
    "Les marées sont causées principalement par", "Le joueur a marqué un but à la dernière minute",
    "La langue française comporte de nombreux mots", "The city of Boston is known for",
    "In physics, the second law of thermodynamics states that", "Yesterday afternoon, the committee decided to",
    "To bake a loaf of bread, you first need to", "The history of the Roman Empire begins with",
    "Machine learning models are trained by", "The dog ran across the field and",
    "According to a recent study published in Nature,", "Photosynthesis converts sunlight into",
    "Once upon a time, in a kingdom by the sea,", "The Pythagorean theorem relates the sides of",
    "The weather forecast for tomorrow predicts", "Stock markets around the world fell sharply after",
    "To install the package, run the following command", "Jazz music originated in the early twentieth century",
    "The Amazon River flows through", "My grandmother used to tell me stories about", "Earthquakes occur when tectonic plates",
    "A classic recipe for tomato soup", "In computer science, a hash table is", "Canada is divided into ten provinces, including",
    "The speed of sound in air depends on", "The novel opens with a scene in which", "Honeybees communicate with each other by",
    "The city council approved a new budget", "A black hole is a region of spacetime",
    "On her first day at the new job, Sarah", "To solve a system of linear equations,", "The Eiffel Tower was built for",
    "Ocean currents are driven mainly by", "The team won the championship after",
    "The English language borrowed many words from",
]
# Décisions publiées par B2 (resume_b2.json, T = 0,6, R_h) : indices des prompts multimodaux
B2_MULTIMODAUX = {"Qwen3-1.7B-Base": [8, 21, 30, 33, 35, 36, 38, 42, 51, 52, 61, 62],
                  "SmolLM3-3B-Base": [35, 36, 43, 44, 51, 52, 59, 62, 63]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", default="reproduction_b2.json")
    args = ap.parse_args()

    eng = TrajectoryBifurcationEngine.sgz(args.model, max_new_tokens=args.steps, seed=SEED)
    if eng.tokenizer.pad_token_id is None:                  # comme B2 : remplissage par la fin de texte
        eng.tokenizer.pad_token = eng.tokenizer.eos_token
        eng.pad_id = eng.tokenizer.pad_token_id
    couche = (2 * eng.num_layers) // 3 - 1                  # hidden_states[⌊2L/3⌋] de B2
    t0 = time.time()
    ens = eng.generer_ensemble(PROMPTS, args.n, args.steps, seed_base=SEED + 17)
    R_h = eng.rejouer_moyennes([ens.prompt_ids[p] for p in ens.prompt_of], ens.tokens, [couche])[couche]
    P = len(PROMPTS)
    rep = [R_h[ens.prompt_of == p] for p in range(P)]
    prof = [ens.entropy_tsallis[ens.prompt_of == p] for p in range(P)]
    h = analyser_regimes(rep, seed=SEED)                    # B2 : graine SEED + 100·ti, ti = 0 à T = 0,6
    s = analyser_regimes(prof, seed=SEED + 50)
    label = args.model.split("/")[-1]
    out = {"modele": args.model, "couche_moteur": couche, "minutes": (time.time() - t0) / 60,
           "R_h": {k: v for k, v in h.items() if k in ("pi", "pi_ci", "n_multi", "nul", "puissance", "verdict", "multimodal")},
           "R_S": {k: v for k, v in s.items() if k in ("pi", "pi_ci", "n_multi", "nul", "puissance", "verdict")}}
    print(f"[B2-h] {label} : {h['n_multi']}/{P} prompts multimodaux (π = {h['pi']:.2f}) ; substitut gaussien {h['nul']:.0%} ; "
          f"mélanges {h['puissance']:.0%} → {h['verdict']}")
    print(f"[B2-S] {label} : {s['n_multi']}/{P} ; mélanges {s['puissance']:.0%} → {s['verdict']}")
    if label in B2_MULTIMODAUX:
        ref = np.zeros(P, bool)
        ref[B2_MULTIMODAUX[label]] = True
        same = int((np.array(h["multimodal"]) == ref).sum())
        out["accord_b2"] = same / P
        print(f"[VÉRIF] décisions identiques à B2 : {same}/{P} prompts "
              f"(64/64 attendu avec les mêmes versions de torch et transformers ; un écart de quelques prompts peut venir "
              f"d'arrondis fp16 sur une autre plateforme)")
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
