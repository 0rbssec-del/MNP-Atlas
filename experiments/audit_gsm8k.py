# experiments/audit_gsm8k.py
# Audit des extracteurs de GSM8K : génération et notation automatique (étape 1).
# Plan, critères et prédictions : experiments/AUDIT_GSM8K.md (commits cb9cef7, e0e0fa3 et amendements suivants).
# Extracteurs et métrique copiés de lm-evaluation-harness au commit b733598 :
#   lm_eval/tasks/gsm8k/gsm8k.yaml (regex, group_select, take_first, until, regexes_to_ignore)
#   lm_eval/filters/extraction.py (RegexFilter.apply) et lm_eval/api/metrics.py (exact_match_hf_evaluate).
# Données : GSM8K (openai/gsm8k, licence MIT), copiées dans experiments/data/.
#
# Usage : python experiments/audit_gsm8k.py --out audit_gsm8k.json
#         (essai local : --models Qwen/Qwen3-0.6B --n 6 --lot 3)

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
SEED = 20261105
MODELES = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM3-3B-Base"]
N_PROBLEMES, MAX_NEW = 500, 256

# ---- copie de lm-evaluation-harness @ b733598 --------------------------------------------------------------------
STRICT = r"#### (\-?[0-9\.\,]+)"                 # strict-match, group_select 0, take_first
FLEX = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"            # flexible-extract, group_select -1, take_first
UNTIL = ["Question:", "</s>", "<|im_end|>"]
IGNORE = [",", "\\$", "(?s).*#### ", "\\.$"]     # regexes_to_ignore, puis ignore_case


def regex_filter(resp, pattern, group_select, fallback="[invalid]"):
    """RegexFilter.apply, pour une seule réponse (lm_eval/filters/extraction.py)."""
    if not isinstance(resp, str):
        resp = ""
    match = re.compile(pattern).findall(resp)
    if match:
        match = match[group_select]
        if isinstance(match, tuple):
            match = [m for m in match if m]
            match = match[0] if match else fallback
        match = match.strip()
    else:
        match = fallback
    return match


def exact_match(pred, ref):
    """exact_match_hf_evaluate avec regexes_to_ignore et ignore_case=True (lm_eval/api/metrics.py)."""
    for s in IGNORE:
        pred, ref = re.sub(s, "", pred), re.sub(s, "", ref)
    return pred.lower() == ref.lower()
# -------------------------------------------------------------------------------------------------------------------


def couper(texte):
    """Coupe la suite au premier marqueur d'arrêt (generate_until)."""
    fin = min([texte.find(u) for u in UNTIL if u in texte] or [len(texte)])
    return texte[:fin]


def temoins_extracteurs():
    """20 réponses synthétiques à verdict connu : (réponse, strict attendu, flexible attendu)."""
    cas = [
        ("He has 5 apples.\n#### 5", "5", "5"),
        ("Total = 1,234 dollars.\n#### 1,234", "1,234", "1,234"),
        ("The answer is $18.", "[invalid]", "$18."),
        ("So she earns 3 * 6 = 18 dollars. #### 18", "18", "18"),
        ("No number here.", "[invalid]", "[invalid]"),
        ("#### -7", "-7", "-7"),
        ("It costs $12.50 per hour.", "[invalid]", "$12.50"),
        ("First 4, then 9, finally 25", "[invalid]", "25"),
        ("48/2 = 24 clips. #### 72\nQuestion", "72", "72"),
        ("Answer: 7", "[invalid]", "7"),
        ("#### 3.5", "3.5", "3.5"),
        ("value 10 and 2", "[invalid]", "2"),
        ("We get 100.\n\n#### 100.", "100.", "100."),
        ("ends with $.", "[invalid]", "$."),
        ("#### 42 and 43", "42", "43"),
        ("x = -3", "[invalid]", "-3"),
        ("1000000", "[invalid]", "1000000"),
        ("#### 5\n#### 6", "5", "6"),
        ("so 7", "[invalid]", "7"),
        ("costs 2, 3", "[invalid]", "3"),
    ]
    ok = 0
    for r, s, f in cas:
        gs, gf = regex_filter(r, STRICT, 0), regex_filter(r, FLEX, -1)
        if gs == s and gf == f:
            ok += 1
        else:
            print(f"[TÉMOIN EXTRACTEURS] écart sur {r!r} : strict {gs!r} (attendu {s!r}), flexible {gf!r} (attendu {f!r})")
    print(f"[TÉMOIN EXTRACTEURS] {ok}/{len(cas)} conformes")
    assert exact_match("72", "Natalia sold... #### 72") and exact_match("1,234", "#### 1234")
    assert exact_match("$18.", "#### 18") and not exact_match("[invalid]", "#### 18")
    return ok == len(cas)


def prompt_fewshot(fewshot, question):
    tete = "".join(f"Question: {d['question']}\nAnswer: {d['answer']}\n\n" for d in fewshot)
    return tete + f"Question: {question}\nAnswer:"


@torch.no_grad()
def generer(modele, tok, prompts, lot, dev):
    sorties = []
    for b in range(0, len(prompts), lot):
        enc = tok(prompts[b:b + lot], return_tensors="pt", padding=True).to(dev)
        out = modele.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                              top_k=None, pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            sorties.append(couper(tok.decode(row, skip_special_tokens=False)))
    return sorties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELES)
    ap.add_argument("--n", type=int, default=N_PROBLEMES)
    ap.add_argument("--lot", type=int, default=32)
    ap.add_argument("--out", default="audit_gsm8k.json")
    args = ap.parse_args()
    t0 = time.time()
    if not temoins_extracteurs():
        sys.exit("[ARRÊT] extracteurs non conformes à la source")
    test = json.loads((DATA / "gsm8k_test.json").read_text())
    fewshot = json.loads((DATA / "gsm8k_train_fewshot5.json").read_text())
    idx = sorted(np.random.default_rng(SEED).choice(len(test), size=min(args.n, len(test)), replace=False).tolist())
    prompts = [prompt_fewshot(fewshot, test[i]["question"]) for i in idx]
    print(f"[DONNÉES] {len(idx)} problèmes ; empreinte des indices {hashlib.md5(str(idx).encode()).hexdigest()}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    res = {"setup": {"seed": SEED, "n": len(idx), "indices": idx, "max_new_tokens": MAX_NEW, "until": UNTIL,
                     "lm_eval_commit": "b733598", "torch": torch.__version__, "dtype": str(dt)}, "modeles": {}}
    for mid in args.models:
        t1 = time.time()
        tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        modele = AutoModelForCausalLM.from_pretrained(mid, dtype=dt).to(dev).eval()
        sorties = generer(modele, tok, prompts, args.lot, dev)
        k = min(50, len(prompts))
        relance = generer(modele, tok, prompts[:k], args.lot, dev)
        identiques = float(np.mean([a == b for a, b in zip(sorties[:k], relance)]))
        lignes = []
        for i, s in zip(idx, sorties):
            st, fl = regex_filter(s, STRICT, 0), regex_filter(s, FLEX, -1)
            lignes.append({"i": i, "reponse": s, "strict": st, "flexible": fl, "or": test[i]["answer"],
                           "ok_strict": exact_match(st, test[i]["answer"]), "ok_flexible": exact_match(fl, test[i]["answer"])})
        a_s = np.mean([r["ok_strict"] for r in lignes]); a_f = np.mean([r["ok_flexible"] for r in lignes])
        desaccord = np.mean([r["ok_strict"] != r["ok_flexible"] for r in lignes])
        print(f"[{mid}] exactitude strict {a_s:.3f}, flexible {a_f:.3f} ; désaccord {desaccord:.3f} ; "
              f"relance identique {identiques:.0%} sur {k} ; {(time.time() - t1) / 60:.1f} min", flush=True)
        res["modeles"][mid] = {"lignes": lignes, "exactitude_strict": float(a_s), "exactitude_flexible": float(a_f),
                               "desaccord": float(desaccord), "relance_identique": identiques}
        del modele
        if dev == "cuda":
            torch.cuda.empty_cache()
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False))
    res["setup"]["minutes"] = (time.time() - t0) / 60
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False))
    print(f"[SORTIE] {args.out} ({res['setup']['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
