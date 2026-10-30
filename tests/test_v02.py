# Tests des ajouts de v0.2 : échantillonneur Gumbel identique aux notebooks SGZ, génération par lots,
# représentations, régimes (B2), bifurcation par préfixe, valeur ajoutée sur les jetons, choc (A1).
# Petit Qwen3 à poids aléatoires (CPU) ; seul le tokenizer de Qwen/Qwen3-0.6B est téléchargé.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trajectory_bifurcation_core as core  # noqa: E402
from trajectory_bifurcation_core import TrajectoryBifurcationEngine  # noqa: E402

transformers = pytest.importorskip("transformers")
pytest.importorskip("sklearn")
pytest.importorskip("diptest")

PROMPTS = ["La ville de Lyon est située au confluent", "The city of Boston is known for",
           "Il était une fois, dans un petit village de montagne,"]


@pytest.fixture(scope="module")
def eng():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # pas de réseau ni de cache
        pytest.skip(f"Tokenizer Qwen3 indisponible : {e}")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    model = transformers.Qwen3ForCausalLM(cfg).float()
    with torch.no_grad():
        model.lm_head.weight.mul_(8.0)
    return TrajectoryBifurcationEngine.sgz("toy-Base", model=model, tokenizer=tok, device="cpu", max_new_tokens=10,
                                           layers_to_track=[0, 2, 3], batch_size=4, seed=20261029)


def sgz_gen_sample(model, tok, ctxs, prompt_of, T, nsteps, step0, base):
    """Copie de Gen.sample (notebooks SGZ A1c–B2), référence indépendante du moteur."""
    V = model.get_output_embeddings().weight.shape[0]
    forbid = set(tok.all_special_ids or []) | {tok.eos_token_id}
    ge = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    forbid |= set(ge if isinstance(ge, (list, tuple)) else ([ge] if ge is not None else []))
    forbid.discard(None)
    forbid, n_tok = sorted(forbid), len(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    n = len(ctxs)
    L = max(len(s) for s in ctxs)
    ids = torch.full((n, L), pad, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for j, s in enumerate(ctxs):
        ids[j, L - len(s):] = torch.tensor(s, dtype=torch.long)
        attn[j, L - len(s):] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    S, tk = np.zeros((n, nsteps), np.float32), np.zeros((n, nsteps), np.int64)
    past = None
    with torch.no_grad():
        for s in range(nsteps):
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            z = out.logits[:, -1, :].float()
            S[:, s] = core.entropies_from_logits(z, [1.45])[1.45].numpy()
            seeds = [base + 100003 * prompt_of[j] + step0 + s for j in range(n)]
            score = z / T + core.gumbel_noise(V, seeds, z.device)
            score[:, forbid] = -float("inf")
            if V > n_tok:
                score[:, n_tok:] = -float("inf")
            nxt = torch.argmax(score, dim=-1)
            tk[:, s] = nxt.numpy()
            ids = nxt[:, None]
            attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype)], dim=1)
            pos = pos[:, -1:] + 1
    return S, tk


def test_lot_identique_a_sgz(eng):
    enc = [list(eng.tokenizer.encode(p)) for p in PROMPTS]
    rows = [0 * 256 + 0, 1 * 256 + 3, 2 * 256 + 7]
    S, tk = sgz_gen_sample(eng.model, eng.tokenizer, enc, rows, 0.6, 12, 0, 20261029 + 17)
    o = eng.generer_lot(enc, rows, 12, seed_base=20261029 + 17)
    np.testing.assert_array_equal(o["tokens"], tk)
    np.testing.assert_allclose(o["entropy_tsallis"], S, atol=1e-6)


def test_ensemble_identique_a_sgz_b2(eng):
    """Même ordre des lots (p, j), mêmes identifiants p·256 + j, même base : mêmes jetons que B2."""
    n, T, base = 5, 8, 20261029 + 17
    ens = eng.generer_ensemble(PROMPTS, n, T, seed_base=base, progress=False)
    enc = [list(eng.tokenizer.encode(p)) for p in PROMPTS]
    jobs = [(p, j) for p in range(len(PROMPTS)) for j in range(n)]
    ref = []
    for b0 in range(0, len(jobs), eng.batch_size):
        lot = jobs[b0:b0 + eng.batch_size]
        ref.append(sgz_gen_sample(eng.model, eng.tokenizer, [enc[p] for p, _ in lot], [p * 256 + j for p, j in lot],
                                  0.6, T, 0, base)[1])
    np.testing.assert_array_equal(ens.tokens, np.concatenate(ref))
    assert (ens.prompt_of == np.repeat(np.arange(len(PROMPTS)), n)).all()


def test_remplissage_a_gauche_sans_effet(eng):
    enc = [list(eng.tokenizer.encode(p)) for p in PROMPTS]
    seul = [eng.generer_lot([e], [i], 10, seed_base=7)["tokens"][0] for i, e in enumerate(enc)]
    lot = eng.generer_lot(enc, [0, 1, 2], 10, seed_base=7)["tokens"]
    assert np.mean(np.stack(seul) == lot) >= 0.95


def test_moyennes_egales_a_la_passe_rejouee(eng):
    enc = [list(eng.tokenizer.encode(p)) for p in PROMPTS]
    o = eng.generer_lot(enc, [0, 1, 2], 9, seed_base=3, mean_layers=[1, 2])
    rej = eng.rejouer_moyennes(enc, o["tokens"], [1, 2])
    for l in (1, 2):
        np.testing.assert_allclose(o["hidden_mean"][l], rej[l], atol=1e-4)


def test_moyennes_egales_a_hidden_states_hf(eng):
    """Couche l du moteur = hidden_states[l + 1] de HF (représentation R_h de B2)."""
    enc = [list(eng.tokenizer.encode(PROMPTS[0]))]
    o = eng.generer_lot(enc, [0], 6, seed_base=3, mean_layers=[1])
    seq = torch.tensor([enc[0] + o["tokens"][0].tolist()])
    with torch.no_grad():
        hs = eng.model.model(input_ids=seq, output_hidden_states=True).hidden_states
    np.testing.assert_allclose(o["hidden_mean"][1][0], hs[2][0, -6:].float().mean(0).numpy(), atol=1e-4)


def test_etats_par_pas_comme_api_trajectoire(eng):
    enc = list(eng.tokenizer.encode(PROMPTS[1]))
    o = eng.generer_lot([enc], [eng.seed + 0], 7, seed_base=eng.seed, layers=[2], keep_steps=True)
    tr = eng._generate_one_trajectory(torch.tensor([enc]), seed=eng.seed + 0)
    assert tr.generated_ids[:7] == o["tokens"][0].tolist()
    np.testing.assert_allclose(np.stack(tr.hidden_states[2][:7]), o["hidden_steps"][2][0].astype(np.float32), atol=2e-2)


def test_entropies():
    z = torch.tensor([[0.0, 1.0, 2.0, -1.0]])
    p = torch.softmax(z.double(), -1).numpy()[0]
    e = core.entropies_from_logits(z, [1.0, 1.45])
    assert e[1.0].item() == pytest.approx(-(p * np.log(p)).sum(), rel=1e-12)
    assert e[1.45].item() == pytest.approx((1 - (p ** 1.45).sum()) / 0.45, rel=1e-12)


def test_regimes_calibration_et_puissance():
    rng = np.random.default_rng(0)
    gauss = [rng.normal(0, 1, (160, 32)) * np.exp(rng.normal(0, 1, 32)) for _ in range(8)]
    r = core.analyser_regimes(gauss, seed=1)
    assert r["n_multi"] == 0 and r["nul"] == 0.0
    u = rng.normal(0, 1, 32)
    u /= np.linalg.norm(u)

    def melange():
        X = rng.normal(0, 1, (160, 32))
        X[:80] += 3 * u
        X[80:] -= 3 * u
        return X
    r2 = core.analyser_regimes([melange() for _ in range(8)], seed=1)
    assert r2["pi"] >= 0.75 and r2["verdict"] == "BASSINS"
    assert core.regle_regimes(0.05, 0.0, 0.5) == "NON INTERPRÉTABLE (test sans puissance)"
    assert core.regle_regimes(0.3, 0.0, 0.9) == "MIXTE"


def test_bifurcation_prefixe_plantee():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 2, 120)
    TK = rng.integers(0, 50, (120, 10))
    TK[:, 2] = 100 + lab                         # le 3e jeton décide du régime
    r = core.bifurcation_prefixe(TK[:, :2].copy(), lab, seed=0)
    assert r["k"] is None
    TK[:, :2] = 7                                # préfixe commun, puis bifurcation au 3e jeton
    r = core.bifurcation_prefixe(TK, lab, S=np.ones((120, 10)), seed=0)
    assert r["k"] == 3 and r["gain_max"] > 0.4


def test_valeur_ajoutee_sur_jetons():
    rng = np.random.default_rng(0)
    n, d = 300, 24
    y = np.tile([0, 1], n // 2)
    pref = [list(rng.integers(0, 500, 6)) for _ in range(n)]
    H = rng.normal(0, 1, (n, d))
    H[:, 0] += 2.0 * y                          # l'état porte l'issue, pas les jetons
    r = core.valeur_ajoutee_sur_jetons(H, pref, y, B=200)
    assert r["verdict"] == "VALEUR AJOUTÉE" and r["auc_jetons"] < 0.65
    pref2 = [p[:-1] + [1000 + int(t)] for p, t in zip(pref, y)]   # le dernier jeton porte l'issue
    r2 = core.valeur_ajoutee_sur_jetons(rng.normal(0, 1, (n, d)), pref2, y, B=200)
    assert r2["verdict"] == "MOINS QUE LES JETONS"
    assert core.valeur_ajoutee_sur_jetons(H, pref, np.zeros(n, int))["verdict"].startswith("NON TESTABLE")


def test_choc_rang_et_couplage(eng):
    res = eng.ensemble_choc(PROMPTS, t0=4, n_steps=8, rang_lo=20, rang_hi=40, seed_base=11)
    assert (res["temoin"][:, :4] == res["choc"][:, :4]).all()
    assert (res["temoin"][:, 4] != res["choc"][:, 4]).all()
    assert (res["X"] == res["choc"][:, 4]).all()
    assert ((res["rang"] >= 20) & (res["rang"] <= 41)).all() and (res["surprise"] > 0).all()


def test_classement_detaille(eng):
    mk = lambda ids: core.Trajectory(token_ids=[1, 2] + ids, text="", prompt_len=2)
    a, b = mk([5, 6, 7, 8]), mk([5, 9, 7, 8])
    assert eng.classify_pair_detail(a, b, 3) == "informative_context"
    assert eng.classify_pair_detail(a, mk([5, 9, 3, 8]), 3) == "informative_token"
    assert eng.classify_pair_detail(a, mk([5, 6, 3, 8]), 3) == "positive_control"
    assert eng.classify_pair_detail(a, a, 3) == "negative_control"


def test_recovery_none_si_denominateur_petit(eng):
    tr = eng.generate_ensemble("Raconte une histoire courte.", 2, seed=0)
    eff = eng.forced_patch_effect(tr[0], tr[0], 3, 2, horizon=4)
    assert eff["recovery_multistep"] is None and eff["recovery_loglik"] is None


def test_temoin_negatif_gumbel_exact(eng):
    """Préfixes identiques : patch nul, même bruit → continuations identiques (pas d'arrondi de torch.multinomial)."""
    tr = eng.generate_ensemble("Raconte une histoire courte.", 6, seed=0)
    res = eng.reversibility_analysis(tr, t_patch=0, n_pairs=2, n_controls=3, max_continuation_tokens=6, forced_horizon=0)
    assert res and all(r.final_token_overlap == 1.0 and r.first_divergence_step is None for r in res)


def test_cartographier(eng):
    out = eng.cartographier(PROMPTS[:2], n=24, n_steps=8, layer=2, t_valeur=3)
    assert set(out) == {"couche", "t_valeur", "regimes", "fiches"} and len(out["fiches"]) == 2
    assert out["regimes"]["verdict"]
    import json
    json.dumps(out, default=float)


def test_pas_gardes_et_longueurs(eng):
    enc = [list(eng.tokenizer.encode(p)) for p in PROMPTS]
    tout = eng.generer_lot(enc, [0, 1, 2], 8, seed_base=5, layers=[2], keep_steps=True)
    deux = eng.generer_lot(enc, [0, 1, 2], 8, seed_base=5, layers=[2], keep_steps=True, steps_to_keep=[3, 6])
    np.testing.assert_array_equal(deux["hidden_steps"][2], tout["hidden_steps"][2][:, [3, 6]])
    assert (tout["lengths"] == 8).all()              # jetons spéciaux interdits : longueur fixe
    ens = eng.generer_ensemble(PROMPTS[:1], 4, 6, seed_base=5, layers=[2], keep_steps=True, steps_to_keep=[2],
                               progress=False)
    assert ens.kept_steps == [2] and ens.hidden_steps[2].shape[:2] == (4, 1) and (ens.lengths == 6).all()
