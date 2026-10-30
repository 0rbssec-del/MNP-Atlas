# Tests de l'atlas v0.3, gardés en v0.4 (atlas/bifurcation_atlas.py) : résumés, compatibilité
# avec les entrées v0.1.x, provenance, sérialisation, construction en lot sur
# le moteur v0.2 (petit Qwen3 aléatoire, CPU).

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas import compat  # noqa: E402
from atlas.bifurcation_atlas import BifurcationAtlas  # noqa: E402
from bifurcation_atlas import BifurcationAtlas as Raccourci  # noqa: E402
from trajectory_bifurcation_core import BifurcationResult, PatchResult  # noqa: E402


def paire(kind="informative", detail=None, recovery=None, multi=None, loglik=None, same=None):
    p = PatchResult(
        source_traj_idx=0, target_traj_idx=1, layer=2, t_patch=5,
        original_continuation="a", patched_continuation="b",
        final_token_overlap=0.5, success=True, pair_kind=kind,
        recovery=recovery, recovery_multistep=multi, recovery_loglik=loglik,
        last_token_same=same,
        pair_detail=detail or (kind if kind != "informative" else "informative_token"),
    )
    return p


def entree(prompt="p", patches=(), corr=None, t_js=4, js=None, frac=None):
    summary = {"mean_entropy": 1.5, "t_patch_used": 6}
    if corr is not None:
        summary["corr_js_divergence_jetons"] = corr
    return BifurcationResult(
        prompt=prompt, n_trajectories=8, residual_dist_over_time={},
        js_over_time=np.array(js if js is not None else [0.0, 0.1, 0.3]),
        bifurcation_times={"js": t_js, "residual": None},
        patch_results=list(patches), summary=summary,
        frac_pairs_same_prefix=None if frac is None else np.array(frac),
    )


def test_raccourci_d_import():
    assert Raccourci is BifurcationAtlas


def test_resume_garde_le_sens_v02_et_ajoute_v02_moteur():
    r = entree(patches=[
        paire(detail="informative_token", recovery=1.0, multi=0.9, loglik=0.8),
        paire(detail="informative_token", recovery=-2.0, multi=0.7, loglik=0.6),
        paire(detail="informative_context", recovery=0.5, multi=0.1, loglik=0.2),
        paire(kind="negative_control"),
    ], corr=0.95)
    s = BifurcationAtlas.summarize_entry(r)
    assert s.mean_recovery_informative == pytest.approx((1.0 - 2.0 + 0.5) / 3)   # moyenne, comme v0.2
    assert s.n_informative_patches == 3 and s.patch_success_rate == 1.0
    assert s.n_informative_context == 1 and s.n_informative_token == 2
    assert s.median_recovery_multistep_token == pytest.approx(0.8)
    assert s.median_recovery_loglik_context == pytest.approx(0.2)
    assert s.bifurcation_suit_jetons is True and s.t_patch == 6


def test_correlation_recalculee_pour_entree_v01():
    js = [0.0, 0.0, 0.2, 0.4, 0.5]
    frac = [1.0, 1.0, 0.7, 0.4, 0.3]
    s = BifurcationAtlas.summarize_entry(entree(js=js, frac=frac))
    assert s.corr_js_divergence_jetons == pytest.approx(np.corrcoef(js, 1 - np.array(frac))[0, 1])


def test_paire_picklee_avant_v02():
    """Sans pair_detail dans l'instance, le type vient de last_token_same, pas
    de la valeur par défaut de la classe."""
    p_ctx, p_inconnu, p_neg = paire(same=True), paire(), paire(kind="negative_control")
    for p in (p_ctx, p_inconnu, p_neg):
        del p.__dict__["pair_detail"]
    del p_inconnu.__dict__["last_token_same"]
    assert compat.type_detaille(p_ctx) == "informative_context"
    assert compat.type_detaille(p_inconnu) == "informative"
    assert compat.type_detaille(p_neg) == "negative_control"


def test_classement_mesure_type_agregat():
    a = entree("a", [paire(recovery=0.2, multi=0.9), paire(recovery=0.4, multi=-50.0)])
    b = entree("b", [paire(recovery=0.9, multi=0.5), paire(recovery=0.1, multi=0.6)])
    atlas = BifurcationAtlas()
    atlas.add_many([a, b])
    assert [e.prompt for _, e in atlas.rank_by_recovery()] == ["b", "a"]          # défaut v0.2
    assert [e.prompt for _, e in atlas.rank_by_recovery(mesure="recovery_multistep")] == ["b", "a"]
    med = atlas.rank_by_recovery(mesure="recovery_multistep", agregat="median")
    assert med[0][0] == pytest.approx(0.55)
    assert atlas.get_by_recovery(min_recovery=0.4) == [b]
    with pytest.raises(ValueError):
        atlas.rank_by_recovery(mesure="success")


def test_features_nan_et_recovery_negatif():
    atlas = BifurcationAtlas()
    f = atlas.to_navigation_features(entree(t_js=None, patches=[paire(recovery=-1.0)]))
    assert math.isnan(f["t_star_js"]) and f["mean_recovery"] == -1.0
    f2 = atlas.to_navigation_features(entree())
    assert math.isnan(f2["mean_recovery"])


def test_provenance_comparaison_et_sauvegarde(tmp_path):
    atlas = BifurcationAtlas("t")
    reg = {"protocole": "sgz", "modele": "m", "sampler": "gumbel", "temperature": 0.6,
           "top_p": 1.0, "chat": False, "max_new_tokens": 32}
    atlas.add(entree("a", corr=0.5), reg)
    atlas.add(entree("b"), dict(reg, temperature=0.8, protocole="autre"))
    atlas.add(entree("c"))
    assert atlas.compare_entries(0, 1)["comparable"] is False
    assert atlas.compare_entries(0, 2)["comparable"] is None
    s = atlas.summary()
    assert s["protocoles_melanges"] and s["protocoles"] == {"sgz": 1, "autre": 1, "inconnu": 1}

    atlas.save(tmp_path / "x")
    with pytest.raises(FileExistsError):
        atlas.save(tmp_path / "x")
    lu = BifurcationAtlas.load(tmp_path / "x")
    assert len(lu) == 3 and lu.provenance[0]["protocole"] == "sgz" and lu.provenance[2] is None
    assert lu.get_by_prompt("b").prompt == "b"
    assert lu.metadata["atlas_version"] == "0.4" and lu.metadata["core_version"] == compat.version_moteur()


def test_atlas_v02_sans_provenance(tmp_path):
    atlas = BifurcationAtlas()
    atlas.add(entree("a"))
    atlas.save(tmp_path / "v02")
    meta = (tmp_path / "v02" / "atlas_meta.json")
    import json
    m = json.loads(meta.read_text())
    del m["provenance"]
    meta.write_text(json.dumps(m))
    assert BifurcationAtlas.load(tmp_path / "v02").provenance == [None]


def test_version_moteur():
    assert compat.version_moteur() == "0.2.1"


def test_construction_en_lot_protocole_sgz():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:
        pytest.skip(f"Tokenizer Qwen3 indisponible : {e}")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
    )
    model = transformers.Qwen3ForCausalLM(cfg).float()
    with torch.no_grad():
        model.lm_head.weight.mul_(20.0)
    atlas = BifurcationAtlas.build_from_prompts(
        "sgz", ["Bonjour, je", "Le chat"], n_trajectories=6, n_patch_pairs=2,
        moteur_kwargs=dict(model=model, tokenizer=tok, device="cpu", model_name="toy",
                           max_new_tokens=10, layers_to_track=[0, 2, 3]),
        forced_horizon=4, require_bifurcation=False,
    )
    assert len(atlas) == 2
    assert all(p["protocole"] == "sgz" and p["sampler"] == "gumbel" for p in atlas.provenance)
    s = atlas.summary()
    assert s["protocoles"] == {"sgz": 2} and not s["protocoles_melanges"]
    for e in atlas.entries:
        es = atlas.summarize_entry(e)
        assert es.n_informative_patches == es.n_informative_context + es.n_informative_token
