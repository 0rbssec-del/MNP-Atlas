# Tests de atlas.compat.steer_hook (steering soutenu par ajout) sur un petit Qwen3 à poids aléatoires (CPU).
# Seul le tokenizer de Qwen/Qwen3-0.6B est téléchargé.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas import compat  # noqa: E402

transformers = pytest.importorskip("transformers")

PROMPT = "La ville de Lyon est située au confluent"
T = 10
COUCHE = 2


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
    return compat.construire_moteur("sgz", model=model, tokenizer=tok, device="cpu", model_name="toy",
                                    max_new_tokens=T, layers_to_track=[0, 2, 3], batch_size=8, seed=7)


@pytest.fixture(scope="module")
def ctx(eng):
    return list(eng.tokenizer.encode(PROMPT))


def lot(eng, ctx, rows, **kw):
    return eng.generer_lot([ctx] * len(rows), rows, T, seed_base=11, **kw)["tokens"]


def vecteur(eng, graine=0, norme=30.0):
    v = np.random.default_rng(graine).standard_normal(eng.model.config.hidden_size).astype(np.float32)
    return v * (norme / np.linalg.norm(v))


def test_alpha_zero_est_l_identite(eng, ctx):
    ref = lot(eng, ctx, range(6))
    with compat.steer_hook(eng, COUCHE, vecteur(eng), alpha=0.0):
        assert (lot(eng, ctx, range(6)) == ref).all()
    with compat.steer_hook(eng, COUCHE, np.zeros(64, np.float32), alpha=5.0):
        assert (lot(eng, ctx, range(6)) == ref).all()


def test_steering_change_les_suites_et_le_hook_est_retire(eng, ctx):
    ref = lot(eng, ctx, range(6))
    with compat.steer_hook(eng, COUCHE, vecteur(eng), alpha=1.0):
        steer = lot(eng, ctx, range(6))
    assert (steer != ref).any()
    assert len(eng._decoder_layers()[COUCHE]._forward_hooks) == 0
    assert (lot(eng, ctx, range(6)) == ref).all()


def test_ajout_exact_a_la_derniere_position(eng, ctx):
    ids = torch.tensor([ctx])
    v = vecteur(eng, 1)
    with torch.no_grad(), compat.capture_hooks(eng, [COUCHE]) as c0:
        eng.model(input_ids=ids)
    with torch.no_grad(), compat.steer_hook(eng, COUCHE, v, alpha=0.7), compat.capture_hooks(eng, [COUCHE]) as c1:
        eng.model(input_ids=ids)
    np.testing.assert_allclose((c1[COUCHE] - c0[COUCHE]).numpy(), 0.7 * v, atol=1e-5)


def test_t_debut_respecte(eng, ctx):
    ref = lot(eng, ctx, range(8))
    with compat.steer_hook(eng, COUCHE, vecteur(eng, 2, 60.0), alpha=1.0, t_debut=4):
        steer = lot(eng, ctx, range(8))
    assert (steer[:, :4] == ref[:, :4]).all()          # le jeton k dépend du passage k : avant t_debut, rien ne change
    assert (steer[:, 4:] != ref[:, 4:]).any()


def test_cache_egal_recalcul_complet(eng, ctx):
    """Ajout soutenu avec KV-cache = ajout du vecteur aux positions qui produisent les jetons dans un recalcul complet."""
    t_debut, alpha, v = 2, 1.0, vecteur(eng, 3, 40.0)
    rows = list(range(4))
    with compat.steer_hook(eng, COUCHE, v, alpha=alpha, t_debut=t_debut):
        toks = lot(eng, ctx, rows)
    P = len(ctx)

    def hook(module, inputs, output):
        h, rest = eng._split_output(output)
        h = h.clone()
        h[:, P - 1 + t_debut:P - 1 + T, :] += alpha * torch.as_tensor(v)
        return h if rest is None else (h,) + rest

    ids = torch.tensor([ctx + toks[j].tolist() for j in range(len(rows))])
    handle = eng._decoder_layers()[COUCHE].register_forward_hook(hook)
    try:
        with torch.no_grad():
            z = eng.model(input_ids=ids).logits.float()
    finally:
        handle.remove()
    for s in range(T):
        seeds = [11 + 100003 * rows[j] + s for j in range(len(rows))]
        assert (eng._gumbel_next(z[:, P - 1 + s, :], seeds).numpy() == toks[:, s]).all(), f"pas {s}"


def test_vecteurs_par_ligne(eng, ctx):
    v1, v2 = vecteur(eng, 4), vecteur(eng, 5)
    rows = [0, 1]
    with compat.steer_hook(eng, COUCHE, np.stack([v1, v2]), alpha=1.0):
        mixte = lot(eng, ctx, rows)
    with compat.steer_hook(eng, COUCHE, v1, alpha=1.0):
        a = lot(eng, ctx, rows)
    with compat.steer_hook(eng, COUCHE, v2, alpha=1.0):
        b = lot(eng, ctx, rows)
    assert (mixte[0] == a[0]).all() and (mixte[1] == b[1]).all()
