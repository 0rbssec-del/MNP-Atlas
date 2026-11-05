# experiments/relance_gsm8k.py
# Test de la cause du témoin de relance échoué (Qwen2.5-1.5B, 64 % identique). Plan : experiments/AUDIT_GSM8K.md,
# amendement 3. Réutilise les prompts, le décodage et les extracteurs d'experiments/audit_gsm8k.py.
#
# Usage : python experiments/relance_gsm8k.py --out relance_gsm8k.json
#         (essai local : --model Qwen/Qwen3-0.6B --k 4 --k2 3 --lot 2 --n1 2 --n1b 1 --max-new 24 --sans-ref)
import argparse, gzip, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_gsm8k as A  # noqa: E402

REF = A.ROOT / "experiments" / "resultats" / "audit_gsm8k" / "audit_gsm8k.json.gz"


def noter(sorties, idx, test):
    s = [A.exact_match(A.regex_filter(x, A.STRICT, 0), test[i]["answer"]) for x, i in zip(sorties, idx)]
    f = [A.exact_match(A.regex_filter(x, A.FLEX, -1), test[i]["answer"]) for x, i in zip(sorties, idx)]
    return np.array(s, float), np.array(f, float)


def ic_apparie(d, rng, b=10000):
    m = d[rng.integers(0, len(d), (b, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--k2", type=int, default=50)
    ap.add_argument("--lot", type=int, default=32)
    ap.add_argument("--n1", type=int, default=200)
    ap.add_argument("--n1b", type=int, default=20)
    ap.add_argument("--sans-ref", action="store_true")
    ap.add_argument("--sans-fp32", action="store_true")
    ap.add_argument("--max-new", type=int, default=A.MAX_NEW)  # essais locaux seulement
    ap.add_argument("--out", default="relance_gsm8k.json")
    a = ap.parse_args()
    A.MAX_NEW = a.max_new
    t0 = time.time()
    if not A.temoins_extracteurs():
        sys.exit("[ARRÊT] extracteurs non conformes")
    test = json.loads((A.DATA / "gsm8k_test.json").read_text())
    fewshot = json.loads((A.DATA / "gsm8k_train_fewshot5.json").read_text())
    idx = sorted(np.random.default_rng(A.SEED).choice(len(test), size=A.N_PROBLEMES, replace=False).tolist())
    prompts = [A.prompt_fewshot(fewshot, test[i]["question"]) for i in idx]
    ref = None
    if not a.sans_ref:
        r0 = json.loads(gzip.open(REF).read())
        assert r0["setup"]["indices"] == idx, "indices différents de la passe principale"
        ref = [x["reponse"] for x in r0["modeles"][a.model]["lignes"]]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    res = {"setup": vars(a) | {"torch": torch.__version__, "dev": dev}, "sorties": {}}

    def charger(dt):
        return AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()

    def run(nom, m, p, lot):
        t = time.time()
        s = A.generer(m, tok, p, lot, dev)
        res["sorties"][nom] = s
        print(f"[{nom}] {len(p)} problèmes, lots de {lot} : {(time.time() - t) / 60:.1f} min", flush=True)
        return s

    dt = torch.float16 if dev == "cuda" else torch.float32
    m = charger(dt)
    r1a = run("R1a", m, prompts[:a.k], a.lot)
    r1b = run("R1b", m, prompts[:a.k], a.lot)
    r2 = run("R2", m, prompts[:a.k2], a.lot)
    r3 = run("R3", m, prompts[:a.n1], 1)
    r3b = run("R3b", m, prompts[:a.n1b], 1)
    del m
    eq = lambda x, y: [u == v for u, v in zip(x, y)]  # noqa: E731
    c = a.lot  # taille du lot commun
    v = {
        "R1a=R1b": sum(eq(r1a, r1b)), "k": a.k,
        "R2=R1a lot commun": sum(eq(r2[:c], r1a[:c])), "lot commun": min(c, a.k2),
        "R2=R1a lot modifié": sum(eq(r2[c:], r1a[c:a.k2])), "lot modifié": max(0, a.k2 - c),
        "R3b=R3": sum(eq(r3b, r3[:a.n1b])), "n1b": a.n1b,
    }
    if ref:
        v["R1a=passe principale"] = sum(eq(r1a, ref[:a.k]))
        v["R3=passe principale"] = sum(eq(r3, ref[:a.n1])); v["n1"] = a.n1
    print("[COMPARAISONS]", v)
    if v["R1a=R1b"] < a.k:
        verdict = "NON-DÉTERMINISME À LOT FIXE"
    elif v["R2=R1a lot commun"] == v["lot commun"] and v["R2=R1a lot modifié"] < v["lot modifié"]:
        verdict = "COMPOSITION DU LOT"
    else:
        verdict = "INDÉTERMINÉ"
    print(f"[VERDICT CAUSE] {verdict}")
    res["comparaisons"], res["verdict_cause"] = v, verdict
    if ref:
        rng = np.random.default_rng(0)
        s3, f3 = noter(r3, idx[:a.n1], test)
        s0, f0 = noter(ref[:a.n1], idx[:a.n1], test)
        eff = {}
        for nom, x, y in [("strict", s3, s0), ("flexible", f3, f0)]:
            d, lo, hi = ic_apparie(x - y, rng)
            eff[nom] = {"lot1": float(x.mean()), "lot32": float(y.mean()), "diff": d, "ic95": [lo, hi],
                        "verdict": "EFFET DÉTECTÉ" if lo > 0 or hi < 0 else f"effet non détecté à n = {a.n1}"}
            print(f"[EFFET SUR LE SCORE] {nom} : lots de 1 {x.mean():.3f}, passe principale {y.mean():.3f}, "
                  f"diff {d:+.3f} [{lo:+.3f} ; {hi:+.3f}] → {eff[nom]['verdict']}")
        res["effet_score"] = eff
    if not a.sans_fp32:
        m = charger(torch.float32)
        q1 = run("R4_R1a_fp32", m, prompts[:a.k], a.lot)
        q2 = run("R4_R2_fp32", m, prompts[:a.k2], a.lot)
        del m
        w = {"commun": sum(eq(q2[:c], q1[:c])), "modifié": sum(eq(q2[c:], q1[c:a.k2])), "lot modifié": max(0, a.k2 - c),
             "fp32=fp16 (R1a)": sum(eq(q1, r1a))}
        print("[SECONDAIRE float32]", w)
        res["float32"] = w
    res["setup"]["minutes"] = (time.time() - t0) / 60
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
    print(f"[SORTIE] {a.out} ({res['setup']['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
