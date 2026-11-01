# Tests de experiments/dose_reponse_orthogonal.py : orthogonalisation, logique du verdict, et un passage complet
# fit → eval → contrôles orthogonaux sur un petit Qwen3 aléatoire (CPU).

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import dose_reponse as dr  # noqa: E402
import dose_reponse_orthogonal as do  # noqa: E402
from atlas import compat  # noqa: E402

transformers = pytest.importorskip("transformers")
pytest.importorskip("sklearn")
pytest.importorskip("diptest")

COUCHE = 2


def test_orthogonaliser():
    rng = np.random.default_rng(0)
    d, v = rng.standard_normal(50) * 3, rng.standard_normal(50)
    w = do.orthogonaliser(v, d)
    assert abs(do.cosinus(w, d)) < 1e-12
    assert np.linalg.norm(w) == pytest.approx(np.linalg.norm(d))
    assert abs(do.cosinus(do.orthogonaliser(2.5 * d + 0.1 * v, d), d)) < 1e-12      # retire bien la composante le long de d
    # une direction déjà orthogonale ne change que d'échelle
    u = v - (v @ d) / (d @ d) * d
    assert do.cosinus(do.orthogonaliser(u, d), u) == pytest.approx(1.0)


def faux(n, delta_vraie, delta_orth, dnll=0.1, cos=1e-9):
    res = {"fit": {"modele": "toy", "couche": 2}, "prompts": {}}
    orth = {"prompts": {}}
    for i in range(n):
        res["prompts"][str(i)] = {"idx": i, "conds": {"vraie@1": {"delta": delta_vraie, "delta_nll": dnll},
                                                       "vraie@2": {"delta": 0.4, "delta_nll": 0.6},
                                                       "permute@1": {"delta": 0.1}, "autre@1": {"delta": 0.0}}}
        orth["prompts"][str(i)] = {"ref_identique": True, "conds": {
            f"{c}_orth@{a}": {"delta": x, "delta_nll": 0.1, "cos_avec_d": cos}
            for c, x in (("permute", delta_orth), ("autre", delta_orth)) for a in (1, 2)}}
    return res, orth


def test_verdicts():
    assert do.verdict_orth(*faux(8, 0.3, 0.0))["verdict"] == "CONFIRMÉ"
    assert do.verdict_orth(*faux(8, 0.3, 0.3))["verdict"] == "RÉFUTÉ"           # le nul propre bouge autant : pas d'effet propre
    assert do.verdict_orth(*faux(3, 0.3, 0.0))["verdict"] == "INDÉTERMINÉ"
    assert do.verdict_orth(*faux(8, 0.3, 0.0, cos=0.1))["verdict"] == "PARTIEL"  # contrôles mal construits (O-0)
    assert do.verdict_orth(*faux(8, 0.3, 0.0, dnll=2.0))["verdict"] == "INDÉTERMINÉ"
    v = do.verdict_orth(*faux(8, 0.35, 0.2))                                     # D = 0,15 passe O-A, mais le nul bouge (O-B)
    assert v["O_A"] and not v["O_B"] and v["verdict"] == "PARTIEL"
    assert "VERDICT (contrôles orthogonaux) : CONFIRMÉ" in do.rapport(*faux(8, 0.3, 0.0))


@pytest.fixture(scope="module")
def eng():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:
        pytest.skip(f"Tokenizer Qwen3 indisponible : {e}")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    model = transformers.Qwen3ForCausalLM(cfg).float()
    with torch.no_grad():
        model.lm_head.weight.mul_(8.0)
    return compat.construire_moteur("sgz", model=model, tokenizer=tok, device="cpu", model_name="toy",
                                    max_new_tokens=6, layers_to_track=[COUCHE], batch_size=8, seed=dr.SEED)


def test_passage_complet(eng, tmp_path):
    args = Namespace(out_dir=str(tmp_path), n_fit=24, n_eval=8, n_steps=6, t_debut=0, prompts=[0, 1, 2],
                     selection="forcee", min_rare=2, model="toy", dtype="float32", layer=COUCHE)
    info = dr.phase_fit(eng, args, COUCHE)
    info, donnees = dr.charger(tmp_path)
    res = dr.phase_eval(eng, args, info, donnees)
    orth = do.phase(eng, args, info, donnees, res)
    assert set(orth["prompts"]) == {"0", "1", "2"}
    for o in orth["prompts"].values():
        assert o["ref_identique"]                                    # même machine, mêmes graines : référence identique
        assert set(o["conds"]) == {f"{c}_orth@{a}" for c in do.CONTROLES_ORTH for a in (1, 2)}
        assert all(abs(c["cos_avec_d"]) < 1e-6 and 0 <= c["frac_modifiees"] <= 1 for c in o["conds"].values())
    assert "VERDICT (contrôles orthogonaux)" in do.rapport(res, orth)
    # reprise sans recalcul
    assert do.phase(eng, args, info, donnees, res)["prompts"] == orth["prompts"]
    json.loads((tmp_path / "eval_orth.json").read_text())
