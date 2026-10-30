# experiments/replication_akarlar.py
# Réplication discriminante d'Akarlar (2026), « Hallucination as Trajectory
# Commitment », arXiv 2604.15400 ; code : github.com/akarlaraytu/trajectory-commitment
# (experiments/e07_trajectory_analysis/trajectory_patching.py).
#
# Montage repris de leur code : Qwen2.5-1.5B (base), prompt brut sans BOS,
# T = 0.7 sans top-p, 15 tokens au plus, arrêt sur EOS ou deux retours à la
# ligne dans les 3 derniers tokens, leurs 8 prompts « les plus équilibrés »,
# leur classifieur par sous-chaînes. Différences assumées : transformers au lieu
# de TransformerLens ; bfloat16 sur CPU (AMX) au lieu de float16 sur MPS pour
# les phases échantillonnées ; P1 en float32, pour tester une égalité exacte.
# (Changement de dtype décidé après le test rapide, pour la vitesse, avant tout
# résultat sur P1, P1b ou P2.)
#
# Critères fixés avant exécution :
#   P1  Leur implémentation (patch non persistant, cible = nouveau tirage) au
#       pas 1 : la distribution du pas 1 patchée est celle de la source à TOUTES
#       les couches (JS max < 1e-4). Réfuté si JS > 1e-3 à une couche.
#       Conséquence : leur balayage par couche ne peut pas dépendre de la couche.
#   P1b Taux de bascule échantillonnés avec leur implémentation : pas de
#       différence entre couches (χ² d'homogénéité p > 0.05, deux directions).
#   P2  Patch persistant (KV-cache) sur la trajectoire cible reconstruite :
#       recovery_loglik calculé sur les pas k >= 1 (le pas 0 est récupéré par
#       construction) décroît avec la couche : Spearman ρ(couche, médiane)
#       <= -0.5 dans les deux directions. Réfuté si ρ > 0.
#   Descriptifs : fidélité (taux de bifurcation contre leur tableau 3), taux de
#   base C/H/O, asymétrie rapportée aux taux de base, part des étiquettes
#   « correct » dues à une sous-chaîne dans un autre mot, classement strict.
#
# Usage : python experiments/replication_akarlar.py --prompts akarlar_selected_prompts.json

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.compat import (  # noqa: E402
    PROTOCOLES,
    capture_hooks,
    construire_moteur,
    generateur,
    generer_trajectoire,
    js_exacte,
    patch_hook,
)

MAX_GEN = 15
MAX_GEN_DISCOVERY = 30
TEMPERATURE = PROTOCOLES["akarlar"]["moteur"]["temperature"]   # 0,7, comme leur code
LAYER_SWEEP = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
# Leur tableau 3 (N = 20, 30 tokens) : correct / halluciné / autre
THEIR_COUNTS = {
    20: (10, 10, 0), 24: (8, 10, 2), 10: (8, 7, 5), 16: (7, 8, 5),
    22: (6, 14, 0), 25: (6, 13, 1), 27: (6, 7, 7), 29: (6, 14, 0),
}


# ----------------------------------------------------------------------
# Leur classifieur et leur règle d'arrêt
# ----------------------------------------------------------------------

def classify(text, ground_truth, wrong, strict=False):
    low = text.lower()

    def has(term):
        term = term.lower()
        if strict:
            return re.search(r"\b" + re.escape(term) + r"\b", low) is not None
        return term in low

    if any(has(g) for g in ground_truth):
        return "CORRECT"
    if any(has(w) for w in wrong):
        return "HALLUCINATION"
    return "OTHER"


def stops(tok, ids):
    if ids[-1] == tok.eos_token_id:
        return True
    return len(ids) >= 3 and tok.decode(ids[-3:]).count("\n") >= 2


def truncate(tok, ids):
    for k in range(1, len(ids) + 1):
        if stops(tok, ids[:k]):
            return ids[:k]
    return ids


# ----------------------------------------------------------------------
# Statistiques
# ----------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 3), round(c + h, 3))


def chi2_homogeneity(successes, totals):
    k, n = np.array(successes, float), np.array(totals, float)
    p = k.sum() / n.sum()
    if p in (0.0, 1.0):
        return None, None
    e1, e0 = n * p, n * (1 - p)
    chi2 = float((((k - e1) ** 2) / e1 + (((n - k) - e0) ** 2) / e0).sum())
    dof = len(k) - 1
    pval = float(torch.special.gammaincc(
        torch.tensor(dof / 2, dtype=torch.float64),
        torch.tensor(chi2 / 2, dtype=torch.float64)))
    return chi2, pval


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------
# Génération
# ----------------------------------------------------------------------

class Replication:
    def __init__(self, args):
        self.args = args
        # Protocole « akarlar » : torch.multinomial, T = 0,7, sans top-p, prompt brut
        self.engine = construire_moteur(
            "akarlar",
            model_name=args.model,
            torch_dtype=getattr(torch, args.dtype),
            layers_to_track=None,
            max_new_tokens=MAX_GEN,
            seed=args.seed,
        )
        self.tok = self.engine.tokenizer
        self.model = self.engine.model
        self.n_layers = self.engine.num_layers
        self.engine.layers_to_track = list(range(self.n_layers))
        self.prompts = json.loads(Path(args.prompts).read_text())
        self.results = {"setup": {
            "model": args.model, "temperature": TEMPERATURE,
            "max_gen": MAX_GEN, "layer_sweep": LAYER_SWEEP, "dtype": args.dtype,
            "torch": torch.__version__,
        }}

    def prompt_ids(self, prompt):
        ids = self.tok(prompt, return_tensors="pt")["input_ids"]
        return ids.to(self.engine.device)

    def text(self, ids):
        # Comme leur code : decode sans retirer les tokens spéciaux
        return self.tok.decode(ids)

    def natural(self, prompt_ids, seed, record=True, max_gen=MAX_GEN):
        layers = self.engine.layers_to_track
        if not record:
            self.engine.layers_to_track = []
        self.engine.max_new_tokens = max_gen
        try:
            traj = generer_trajectoire(self.engine, prompt_ids, seed)
        finally:
            self.engine.layers_to_track = layers
            self.engine.max_new_tokens = MAX_GEN
        new = truncate(self.tok, traj.generated_ids)
        n = len(new)
        traj.token_ids = traj.token_ids[: traj.prompt_len + n]
        traj.entropy_shannon = traj.entropy_shannon[:n]
        traj.topk_distributions = traj.topk_distributions[:n]
        traj.hidden_states = {l: v[:n] for l, v in traj.hidden_states.items()}
        traj.text = self.text(new)
        return traj

    @torch.no_grad()
    def their_patch_generation(self, prompt_ids, vector, layer, patch_step, seed):
        """Portage littéral de generate_with_patch : pas de cache, patch de la
        dernière position à un seul passage (pas `patch_step`)."""
        gen = generateur(self.engine, seed)
        ids = prompt_ids.clone()
        out = []
        for step in range(MAX_GEN):
            if step == patch_step:
                with patch_hook(self.engine, layer, ids.shape[1] - 1, vector):
                    logits = self.model(input_ids=ids, use_cache=False).logits
            else:
                logits = self.model(input_ids=ids, use_cache=False).logits
            probs = F.softmax(logits[0, -1].float() / TEMPERATURE, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen).item())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
            if stops(self.tok, out):
                break
        return self.text(out)

    def persistent_generation(self, prefix_ids, seed, patch=None):
        """Patch persistant (KV-cache) ; la règle d'arrêt est appliquée après
        coup, ce qui est équivalent car elle ne dépend que du passé."""
        n_prefix_gen = prefix_ids.shape[1] - self.prompt_len
        cont = self.engine.continue_from_prefix(
            prefix_ids, max_new_tokens=MAX_GEN - n_prefix_gen, seed=seed,
            patch=patch,
        )
        gen = prefix_ids[0, self.prompt_len:].tolist() + cont.token_ids
        return self.text(truncate(self.tok, gen))

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def label(self, text, p):
        return {
            "text": text,
            "label": classify(text, p["ground_truth"], p["wrong_indicators"]),
            "label_strict": classify(
                text, p["ground_truth"], p["wrong_indicators"], strict=True
            ),
        }

    def phase_fidelity_and_pool(self):
        fid, pools = [], {}
        for pi, p in enumerate(self.prompts):
            ids = self.prompt_ids(p["prompt"])
            self.prompt_len = ids.shape[1]
            t0 = time.time()
            disc = [
                self.label(self.natural(ids, 10_000 + 100 * pi + i, record=False,
                                        max_gen=MAX_GEN_DISCOVERY).text, p)
                for i in range(self.args.n_discovery)
            ]
            counts = {c: sum(d["label"] == c for d in disc)
                      for c in ("CORRECT", "HALLUCINATION", "OTHER")}
            fid.append({"idx": p["idx"], "ours": counts,
                        "theirs": THEIR_COUNTS.get(p["idx"]), "samples": disc})

            runs = []
            for i in range(self.args.n_pool):
                tr = self.natural(ids, 20_000 + 100 * pi + i)
                runs.append((tr, dict(self.label(tr.text, p), fill=False)))
            # Comme leur code : tirer jusqu'à 3 runs par classe (100 essais
            # au plus) ; ces tirages supplémentaires sont exclus des taux de base
            attempt = len(runs)
            while attempt < 100 and min(
                sum(r[1]["label"] == c for r in runs)
                for c in ("CORRECT", "HALLUCINATION")
            ) < 3:
                tr = self.natural(ids, 20_000 + 100 * pi + attempt)
                lab = self.label(tr.text, p)
                attempt += 1
                if sum(r[1]["label"] == lab["label"] for r in runs) < 3 and lab["label"] != "OTHER":
                    runs.append((tr, dict(lab, fill=True)))
            pools[p["idx"]] = runs
            print(f"[pool] {p['idx']:2d} découverte {counts} | 15 tokens : "
                  f"{[sum(r[1]['label'] == c for r in runs) for c in ('CORRECT', 'HALLUCINATION', 'OTHER')]}"
                  f" ({time.time() - t0:.0f}s)", flush=True)
        self.pools = pools
        self.results["fidelity"] = fid
        self.results["pool"] = {
            str(k): [dict(r[1], t0=r[0].generated_ids[0]) for r in v]
            for k, v in pools.items()
        }

    def sources(self, idx, n=3):
        runs = self.pools[idx]
        c = [r[0] for r in runs if r[1]["label"] == "CORRECT"][:n]
        h = [r[0] for r in runs if r[1]["label"] == "HALLUCINATION"][:n]
        return c, h

    @torch.no_grad()
    def phase_p1(self, pool):
        """Leur patch au pas 1 sur un nouveau tirage : distribution du pas 1.

        N'utilise que les tokens t0 du pool : les états de la source au pas 1
        sont recalculés dans le dtype courant (float32 pour un test exact).
        """
        rows = []
        layers = list(range(self.n_layers))
        for pi, p in enumerate(self.prompts):
            ids = self.prompt_ids(p["prompt"])
            runs = pool.get(str(p["idx"]), [])
            c = [r for r in runs if r["label"] == "CORRECT"]
            h = [r for r in runs if r["label"] == "HALLUCINATION"]
            if not c or not h:
                continue
            for name, src in (("H->C", c[0]), ("C->H", h[0])):
                src_ids = torch.cat([ids, torch.tensor([[src["t0"]]])], 1)
                with capture_hooks(self.engine, layers) as cap:
                    lg = self.model(input_ids=src_ids, use_cache=False).logits
                p_src = F.softmax(lg[0, -1].double(), -1)
                states = {l: cap[l].clone() for l in layers}
                fresh = list(dict.fromkeys(
                    r["t0"] for r in runs if r["t0"] != src["t0"]))[:3]
                for t0 in fresh:
                    tgt_ids = torch.cat([ids, torch.tensor([[t0]])], 1)
                    p_tgt = F.softmax(self.model(input_ids=tgt_ids).logits[0, -1].double(), -1)
                    before = float(js_exacte(p_tgt[None], p_src[None])[0])
                    for layer in layers:
                        vec = states[layer]
                        with patch_hook(self.engine, layer, tgt_ids.shape[1] - 1, vec):
                            lg = self.model(input_ids=tgt_ids, use_cache=False).logits
                        p_pat = F.softmax(lg[0, -1].double(), -1)
                        rows.append({
                            "idx": p["idx"], "direction": name, "t0": t0,
                            "layer": layer, "js_before": before,
                            "js_after": float(js_exacte(p_pat[None], p_src[None])[0]),
                        })
            print(f"[P1] {p['idx']:2d} fait", flush=True)
        self.results["p1"] = rows

    def phase_p1b(self):
        rows = []
        for pi, p in enumerate(self.prompts):
            ids = self.prompt_ids(p["prompt"])
            c, h = self.sources(p["idx"])
            if not c or not h:
                continue
            t0 = time.time()
            for layer in LAYER_SWEEP:
                for name, src in (("H->C", c[0]), ("C->H", h[0])):
                    if len(src.generated_ids) < 2:
                        continue
                    vec = torch.from_numpy(src.hidden_states[layer][1])
                    for trial in range(self.args.trials):
                        seed = 30_000 + 1000 * pi + 10 * layer + trial + (500 if name == "C->H" else 0)
                        text = self.their_patch_generation(ids, vec, layer, 1, seed)
                        rows.append(dict(self.label(text, p), idx=p["idx"],
                                         direction=name, layer=layer, trial=trial))
            print(f"[P1b] {p['idx']:2d} ({time.time() - t0:.0f}s)", flush=True)
        self.results["p1b"] = rows

    @torch.no_grad()
    def forced_profile(self, src, tgt):
        """recovery_loglik par couche, patch persistant au pas 1 sur la cible
        reconstruite, continuation de la source forcée. Propre et source
        calculés une seule fois."""
        forced = src.generated_ids[1:]
        k = len(forced)
        pos = tgt.prompt_len
        base_t = tgt.token_ids[: tgt.prompt_len + 1]
        base_s = src.token_ids[: src.prompt_len + 1]

        def logp(prefix, patch=None):
            ids = torch.tensor([prefix + forced])
            with torch.no_grad():
                if patch is None:
                    lg = self.model(input_ids=ids, use_cache=False).logits
                else:
                    with patch_hook(self.engine, *patch):
                        lg = self.model(input_ids=ids, use_cache=False).logits
            return F.log_softmax(lg[0, pos: pos + k].double(), -1)

        tok = torch.tensor(forced).unsqueeze(1)
        lp_c, lp_s = logp(base_t), logp(base_s)
        ll_c = lp_c.gather(1, tok).squeeze(1)
        ll_s = lp_s.gather(1, tok).squeeze(1)
        out = []
        for layer in range(self.n_layers):
            vec = torch.from_numpy(src.hidden_states[layer][1])
            lp_p = logp(base_t, (layer, pos, vec))
            ll_p = lp_p.gather(1, tok).squeeze(1)
            js0 = float(js_exacte(lp_p[:1].exp(), lp_s[:1].exp())[0])
            row = {"layer": layer, "js_step0_patched_source": js0}
            for name, sl in (("all", slice(0, k)), ("after", slice(1, k))):
                den = float(ll_s[sl].sum() - ll_c[sl].sum())
                num = float(ll_p[sl].sum() - ll_c[sl].sum())
                row[f"recovery_loglik_{name}"] = num / den if abs(den) > 1e-3 else None
            out.append(row)
        return out

    def phase_p2(self):
        det, samp = [], []
        for pi, p in enumerate(self.prompts):
            ids = self.prompt_ids(p["prompt"])
            self.prompt_len = ids.shape[1]
            c, h = self.sources(p["idx"])
            t0 = time.time()
            for i in range(min(len(c), len(h), self.args.n_pairs)):
                for name, src, tgt in (("H->C", c[i], h[i]), ("C->H", h[i], c[i])):
                    if len(src.generated_ids) < 3 or len(tgt.generated_ids) < 2:
                        continue
                    kind = self.engine.classify_pair(src, tgt, 1)
                    for row in self.forced_profile(src, tgt):
                        det.append(dict(row, idx=p["idx"], direction=name,
                                        pair=i, pair_kind=kind))
                    tgt_prefix = torch.tensor([tgt.token_ids[: tgt.prompt_len + 1]])
                    src_prefix = torch.tensor([src.token_ids[: src.prompt_len + 1]])
                    pos = tgt.prompt_len
                    base_seed = 40_000 + 1000 * pi + 100 * i + (50 if name == "C->H" else 0)
                    for layer in LAYER_SWEEP:
                        vec = torch.from_numpy(src.hidden_states[layer][1])
                        text = self.persistent_generation(
                            tgt_prefix, base_seed + layer, patch=(layer, pos, vec))
                        samp.append(dict(self.label(text, p), idx=p["idx"],
                                         direction=name, pair=i, layer=layer,
                                         condition="patch_persistant"))
                    for ref, prefix in (("sans_patch", tgt_prefix), ("echange_token", src_prefix)):
                        for s in range(2):
                            text = self.persistent_generation(prefix, base_seed + 90 + s)
                            samp.append(dict(self.label(text, p), idx=p["idx"],
                                             direction=name, pair=i, layer=None,
                                             condition=ref))
            print(f"[P2] {p['idx']:2d} ({time.time() - t0:.0f}s)", flush=True)
        self.results["p2_deterministic"] = det
        self.results["p2_sampled"] = samp

    def save(self):
        Path(self.args.out).write_text(
            json.dumps(self.results, indent=1, ensure_ascii=False, default=str))


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------

def rate(rows, direction, key="label"):
    target = "CORRECT" if direction == "H->C" else "HALLUCINATION"
    k = sum(r[key] == target for r in rows)
    return k, len(rows)


def report(res):
    out = []
    say = out.append
    say("=== FIDÉLITÉ (découverte, N par prompt, 30 tokens) ===")
    ours_c, theirs_c = [], []
    for f in res.get("fidelity", []):
        n = sum(f["ours"].values())
        say(f"  prompt {f['idx']:2d} : nous C/H/O = {f['ours']['CORRECT']}/{f['ours']['HALLUCINATION']}/{f['ours']['OTHER']} (n={n}) | eux {f['theirs']}")
        ours_c.append(f["ours"]["CORRECT"] / n)
        theirs_c.append(f["theirs"][0] / 20)
    if ours_c:
        say(f"  taux correct moyen : nous {np.mean(ours_c):.2f}, eux {np.mean(theirs_c):.2f} ; Spearman {spearman(ours_c, theirs_c):+.2f}")

    pool = [r for v in res.get("pool", {}).values() for r in v if not r.get("fill")]
    if pool:
        n = len(pool)
        for key in ("label", "label_strict"):
            c = sum(r[key] == "CORRECT" for r in pool)
            h = sum(r[key] == "HALLUCINATION" for r in pool)
            say(f"=== TAUX DE BASE (15 tokens, sans patch, {key}) : correct {c}/{n} {wilson(c, n)}, halluciné {h}/{n} {wilson(h, n)}")
        sub = sum(r["label"] == "CORRECT" and r["label_strict"] != "CORRECT" for r in pool)
        tot = sum(r["label"] == "CORRECT" for r in pool)
        say(f"  « correct » dû seulement à une sous-chaîne dans un autre mot : {sub}/{tot}")

    p1 = res.get("p1", [])
    if p1:
        say("=== P1 : leur patch au pas 1, distribution du pas 1 vs source ===")
        say(f"  JS avant patch (cible vs source) : médiane {np.median([r['js_before'] for r in p1]):.3f}")
        mx = max(r["js_after"] for r in p1)
        by_layer = {l: max(r["js_after"] for r in p1 if r["layer"] == l) for l in sorted({r['layer'] for r in p1})}
        say(f"  JS après patch : max sur toutes couches/paires = {mx:.2e} (n={len(p1)})")
        say(f"  max par couche : {', '.join(f'L{l}:{v:.1e}' for l, v in by_layer.items())}")
        say(f"  VERDICT P1 : {'CONFIRMÉ' if mx < 1e-4 else ('RÉFUTÉ' if mx > 1e-3 else 'INDÉTERMINÉ')}")

    p1b = res.get("p1b", [])
    if p1b:
        say("=== P1b : leur implémentation échantillonnée (balayage de couches) ===")
        for d in ("H->C", "C->H"):
            ks, ns = [], []
            for l in LAYER_SWEEP:
                k, n = rate([r for r in p1b if r["direction"] == d and r["layer"] == l], d)
                ks.append(k)
                ns.append(n)
            chi2, pv = chi2_homogeneity(ks, ns)
            K, N = sum(ks), sum(ns)
            say(f"  {d} : par couche {[f'{k}/{n}' for k, n in zip(ks, ns)]} ; global {K}/{N} = {K / max(N, 1):.2f} {wilson(K, N)} ; χ² p = {pv}")
            verdict = "INDÉTERMINÉ (aucune variation)" if pv is None else ("CONFIRMÉ" if pv > 0.05 else "NON CONFIRMÉ")
            say(f"     VERDICT P1b {d} : {verdict}")

    det = res.get("p2_deterministic", [])
    if det:
        say("=== P2 : patch persistant, cible reconstruite, recovery_loglik (pas k >= 1) ===")
        for d in ("H->C", "C->H"):
            rows = [r for r in det if r["direction"] == d and r["pair_kind"] != "negative_control"]
            layers = sorted({r["layer"] for r in rows})
            med = [np.median([r["recovery_loglik_after"] for r in rows
                              if r["layer"] == l and r["recovery_loglik_after"] is not None] or [np.nan])
                   for l in layers]
            ok = [(l, m) for l, m in zip(layers, med) if not np.isnan(m)]
            rho = spearman([l for l, _ in ok], [m for _, m in ok]) if len(ok) > 2 else None
            js0 = max(r["js_step0_patched_source"] for r in rows) if rows else None
            say(f"  {d} : n paires = {len({(r['idx'], r['pair']) for r in rows})} ; JS pas 0 patché/source max = {js0:.1e}")
            say(f"     médiane par couche : {', '.join(f'L{l}:{m:+.2f}' for l, m in ok)}")
            verdict = "CONFIRMÉ" if rho is not None and rho <= -0.5 else ("RÉFUTÉ" if rho is not None and rho > 0 else "INDÉTERMINÉ")
            say(f"     Spearman ρ(couche, médiane) = {rho:+.2f} → VERDICT P2 {d} : {verdict}")

    samp = res.get("p2_sampled", [])
    if samp:
        say("=== P2 échantillonné : taux de bascule (patch persistant) et références ===")
        for d in ("H->C", "C->H"):
            line = []
            for cond in ("sans_patch", "echange_token"):
                k, n = rate([r for r in samp if r["direction"] == d and r["condition"] == cond], d)
                line.append(f"{cond} {k}/{n}")
            per_layer = []
            for l in LAYER_SWEEP:
                k, n = rate([r for r in samp if r["direction"] == d and r["layer"] == l
                             and r["condition"] == "patch_persistant"], d)
                per_layer.append(f"L{l}:{k}/{n}")
            say(f"  {d} : {' ; '.join(line)}")
            say(f"     patch persistant : {', '.join(per_layer)}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-discovery", type=int, default=20)
    ap.add_argument("--n-pool", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phases", default="pool,p1b,p2")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--pool-from", default=None,
                    help="JSON d'un run précédent (pour P1 seul, en float32)")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--out", default="replication_akarlar.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rep = Replication(args)
    if args.max_prompts:
        rep.prompts = rep.prompts[: args.max_prompts]
    phases = args.phases.split(",")
    if args.pool_from:
        prev = json.loads(Path(args.pool_from).read_text())
        rep.results.update({k: v for k, v in prev.items() if k != "setup"})
    for name, fn in (("pool", rep.phase_fidelity_and_pool),
                     ("p1", lambda: rep.phase_p1(rep.results["pool"])),
                     ("p1b", rep.phase_p1b), ("p2", rep.phase_p2)):
        if name in phases:
            t0 = time.time()
            fn()
            rep.results.setdefault("timing_s", {})[name] = round(time.time() - t0)
            rep.save()
    print(report(rep.results))


if __name__ == "__main__":
    main()
