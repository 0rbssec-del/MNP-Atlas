# Tests du moteur sur un petit Qwen3 à poids aléatoires (CPU, quelques secondes).
# Seul le tokenizer de Qwen/Qwen3-0.6B est téléchargé (~11 Mo, une fois).

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_bifurcation_core import TrajectoryBifurcationEngine  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def engine():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # pas de réseau ni de cache
        pytest.skip(f"Tokenizer Qwen3 indisponible : {e}")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
    )
    model = transformers.Qwen3ForCausalLM(cfg).float()
    # Poids aléatoires : agrandir les logits pour des distributions non plates
    with torch.no_grad():
        model.lm_head.weight.mul_(20.0)
    return TrajectoryBifurcationEngine(
        model=model,
        tokenizer=tok,
        device="cpu",
        max_new_tokens=12,
        temperature=1.0,
        top_p=0.95,
        topk_for_divergence=64,
    )


@pytest.fixture(scope="module")
def prompt_ids(engine):
    return engine.build_prompt_ids("Raconte une histoire courte.")


@pytest.fixture(scope="module")
def trajectories(engine):
    return engine.generate_ensemble("Raconte une histoire courte.", 6, seed=0)


def test_capture_hooks_match_hidden_states_except_last_layer(engine, prompt_ids):
    layers = list(range(engine.num_layers))
    with torch.no_grad(), engine._capture_hooks(layers) as cap:
        out = engine.model(input_ids=prompt_ids, output_hidden_states=True)
    for l in layers[:-1]:
        torch.testing.assert_close(cap[l], out.hidden_states[l + 1][0, -1])
    # hidden_states[-1] inclut la norme finale : ce n'est pas le point d'injection
    assert not torch.allclose(cap[layers[-1]], out.hidden_states[-1][0, -1])


def test_kv_cache_matches_full_recompute(engine, prompt_ids):
    traj = engine._generate_one_trajectory(prompt_ids, seed=3)
    seq = torch.tensor([traj.token_ids])
    with torch.no_grad():
        logits = engine.model(input_ids=seq).logits[0].float()
    for t, (idx, probs, _) in enumerate(traj.topk_distributions):
        full = torch.softmax(logits[traj.prompt_len - 1 + t], dim=-1)
        np.testing.assert_allclose(full[idx].numpy(), probs, atol=1e-5)


def test_seeded_generation_is_reproducible(engine, prompt_ids):
    a = engine._generate_one_trajectory(prompt_ids, seed=11)
    b = engine._generate_one_trajectory(prompt_ids, seed=11)
    assert a.token_ids == b.token_ids


def test_text_excludes_prompt(engine, trajectories):
    for tr in trajectories:
        assert tr.text == engine.tokenizer.decode(
            tr.generated_ids, skip_special_tokens=True
        )
        assert "Raconte" not in tr.text


def test_patch_persists_beyond_first_step(engine, prompt_ids):
    # Même suite de tokens forcée : le patch doit encore agir aux pas suivants
    layer, pos = 1, prompt_ids.shape[1] - 1
    vec = torch.randn(engine.model.config.hidden_size) * 5
    forced = torch.tensor([[1000, 2000]])

    def step_logits(patch):
        with torch.no_grad(), (
            engine._patch_hook(layer, pos, vec) if patch else _null()
        ):
            out = engine.model(input_ids=prompt_ids, use_cache=True)
            past, logs = out.past_key_values, [out.logits[0, -1]]
            for k in range(forced.shape[1]):
                out = engine.model(
                    input_ids=forced[:, k : k + 1],
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                logs.append(out.logits[0, -1])
        return logs

    clean, patched = step_logits(False), step_logits(True)
    for k in range(len(clean)):
        assert (clean[k] - patched[k]).abs().max() > 1e-3, f"pas {k}"

    # Référence sans cache : patch à `pos` sur chaque passe complète
    def persist_hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        o = o.clone()
        o[:, pos] = vec
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o

    h = engine._decoder_layers()[layer].register_forward_hook(persist_hook)
    try:
        with torch.no_grad():
            seq = torch.cat([prompt_ids, forced], dim=1)
            ref = engine.model(input_ids=seq).logits[0, prompt_ids.shape[1] - 1 :]
    finally:
        h.remove()
    for k in range(len(patched)):
        torch.testing.assert_close(patched[k], ref[k], atol=1e-4, rtol=1e-4)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def test_identity_patch_is_noop(engine, trajectories):
    tr = trajectories[0]
    t, layer = 3, engine.layers_to_track[1]
    prefix = torch.tensor([tr.token_ids[: tr.prompt_len + t]])
    vec = torch.from_numpy(tr.hidden_states[layer][t])
    clean = engine.continue_from_prefix(prefix, 10, seed=5)
    patched = engine.continue_from_prefix(
        prefix, 10, seed=5, patch=(layer, prefix.shape[1] - 1, vec)
    )
    assert clean.token_ids == patched.token_ids


def test_patch_test_on_diverged_pair(engine, trajectories):
    layer = engine.layers_to_track[1]
    t = 6
    pairs = [
        (i, j)
        for i in range(len(trajectories))
        for j in range(len(trajectories))
        if i != j
        and trajectories[i].generated_ids[:t] != trajectories[j].generated_ids[:t]
        and min(len(trajectories[i].entropy_shannon),
                len(trajectories[j].entropy_shannon)) > t
    ]
    assert pairs, "aucune paire divergente avant t"
    i, j = pairs[0]
    r = engine.activation_patch_test(trajectories, i, j, t, layer, 8)
    assert not r.prefixes_identical
    assert r.js_clean_vs_recorded < 1e-6
    assert r.js_patched_target > 0
    assert "Raconte" not in r.original_continuation


def test_controls_in_reversibility_analysis(engine, trajectories):
    # À t_patch = 0, toutes les paires partagent le même préfixe (le prompt)
    results = engine.reversibility_analysis(
        trajectories, t_patch=0, n_pairs=2, n_controls=2,
        max_continuation_tokens=6,
    )
    assert len(results) == 2
    for r in results:
        assert r.pair_kind == "negative_control"
        assert r.prefixes_identical
        assert r.recovery is None
        assert r.final_token_overlap == 1.0
        assert r.first_divergence_step is None


def test_positive_control_recovery_is_one(engine, trajectories):
    # Seul le dernier token du préfixe diffère : le contexte est identique,
    # donc remplacer l'état de la dernière position reproduit la source
    layer = engine.layers_to_track[1]
    for t in range(1, 8):
        for i, src in enumerate(trajectories):
            for j, tgt in enumerate(trajectories):
                if i != j and engine.classify_pair(src, tgt, t) == "positive_control":
                    r = engine.activation_patch_test(trajectories, i, j, t, layer, 4)
                    if r.recovery is not None:
                        assert r.recovery == pytest.approx(1.0, abs=1e-4)
                        return
    pytest.fail("aucun témoin positif trouvé")


def test_js_divergence_properties():
    js = TrajectoryBifurcationEngine.js_divergence_topk
    a = (np.array([1, 2, 3]), np.array([0.5, 0.3, 0.1]), 0.1)
    b = (np.array([3, 4, 5]), np.array([0.6, 0.2, 0.1]), 0.1)
    assert js(a, a) == pytest.approx(0.0, abs=1e-12)
    assert js(a, b) == pytest.approx(js(b, a))
    assert 0 < js(a, b) <= math.log(2) + 1e-9
    disjoint_a = (np.array([1]), np.array([1.0]), 0.0)
    disjoint_b = (np.array([2]), np.array([1.0]), 0.0)
    assert js(disjoint_a, disjoint_b) == pytest.approx(math.log(2), abs=1e-6)


def test_detect_bifurcation_floor_on_zero_baseline(engine):
    series = np.array([0.0, 0.0, 0.0, 0.0, 0.001, 0.002, 0.003])
    div = {"js_over_time": series}
    res = engine.detect_bifurcation(div, measure="js")
    assert res["baseline_degenerate"]
    assert not res["bifurcation_detected"]
    res_old = engine.detect_bifurcation(div, measure="js", abs_floor=0.0)
    assert res_old["t_star"] == 4  # comportement v0.1.3 : détection triviale


def test_analyze_prompt_end_to_end(engine):
    res = engine.analyze_prompt(
        "Raconte une histoire courte.", n_trajectories=4, n_patch_pairs=2
    )
    assert set(res.bifurcation_times) == {"js", "residual"}
    assert len(res.js_over_time) == res.summary["min_len"]
    assert res.frac_pairs_same_prefix[0] == 1.0
