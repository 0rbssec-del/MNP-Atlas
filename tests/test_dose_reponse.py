# Tests de experiments/dose_reponse.py : rejeu non perturbé, classifieur figé, contrôles de même norme, statistiques
# exactes, logique des verdicts, et un passage complet fit → eval → report sur un petit Qwen3 aléatoire (CPU).

import json
import math
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
from atlas import compat  # noqa: E402

transformers = pytest.importorskip("transformers")
pytest.importorskip("sklearn")
pytest.importorskip("diptest")

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
                                    max_new_tokens=6, layers_to_track=[COUCHE], batch_size=8, seed=dr.SEED)


def test_rejouer_egal_au_moteur_et_nll_directe(eng):
    ctx = eng.encoder("La ville de Lyon est située au confluent")
    toks = eng.generer_lot([ctx] * 6, list(range(6)), 6, seed_base=3)["tokens"]
    R, nll, norm = dr.rejouer(eng, ctx, toks, COUCHE)
    np.testing.assert_allclose(R, eng.rejouer_moyennes([ctx] * 6, toks, [COUCHE])[COUCHE], atol=1e-4)
    ids = torch.tensor([ctx + t.tolist() for t in toks])
    with torch.no_grad():
        z = eng.model(input_ids=ids).logits.float()
    lp = torch.log_softmax(z[:, len(ctx) - 1:len(ctx) + 5, :], -1)
    ref = -lp.gather(-1, torch.tensor(toks)[..., None])[..., 0].mean(1).numpy()
    np.testing.assert_allclose(nll, ref, atol=1e-4)
    assert (norm > 0).all()


def donnees_bimodales(n=120, d=40, ecart=6.0, part_rare=0.25, graine=0):
    rng = np.random.default_rng(graine)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    y = (rng.random(n) < part_rare).astype(int)
    X = rng.standard_normal((n, d)) + ecart * np.outer(2 * y - 1, u) * 0.5 + 3.0
    return X, y


def test_classifieur_fige_reproduit_les_etiquettes_et_generalise():
    X, y = donnees_bimodales()
    cl, lab = dr.Classifieur.ajuster(X, seed=1)
    assert (cl.etiqueter(X) == (lab == cl.rare)).mean() >= 0.99
    assert (lab == cl.rare).sum() == min((lab == 0).sum(), (lab == 1).sum())
    # points tirés autour des centres de chaque régime du fit : classés dans leur régime
    rng = np.random.default_rng(5)
    for k in (0, 1):
        pts = X[lab == k][:10].mean(0) + 0.1 * rng.standard_normal((10, X.shape[1]))
        assert (cl.etiqueter(pts) == int(k == cl.rare)).all()


def test_controles_de_meme_norme_et_distincts():
    X, _ = donnees_bimodales()
    cl, lab = dr.Classifieur.ajuster(X, seed=1)
    d = dr.direction_vraie(X, lab, cl.rare)
    autre = np.random.default_rng(9).standard_normal(X.shape[1]) * 7
    c = dr.controles(X, lab, cl.rare, autre, graine=3)
    assert set(c) == {"aleatoire", "permute", "autre"}
    for v in c.values():
        assert np.linalg.norm(v) == pytest.approx(np.linalg.norm(d))
    cos = lambda a, b: abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cos(c["aleatoire"], d) < 0.6 and cos(c["permute"], d) < cos(d, d)
    assert "autre" not in dr.controles(X, lab, cl.rare, None, graine=3)


def test_statistiques_exactes():
    assert dr.wilcoxon_exact([1, 2, 3, 4, 5]) == pytest.approx(1 / 32)
    assert dr.wilcoxon_exact([-1, 2]) == pytest.approx(0.5)
    assert dr.wilcoxon_exact([1, 1, 1]) == pytest.approx(1 / 8)                  # égalités : rangs moyens
    assert dr.wilcoxon_exact([0.0, 0.0]) == 1.0
    assert dr.wilcoxon_exact([-1, -2, -3]) == pytest.approx(1.0)
    assert dr.spearman([0, 0.5, 1, 2], [0.1, 0.2, 0.3, 0.9]) == pytest.approx(1.0)
    assert dr.spearman([0, 0.5, 1, 2], [0.9, 0.3, 0.2, 0.1]) == pytest.approx(-1.0)
    assert math.isnan(dr.spearman([0, 1, 2], [5, 5, 5]))
    assert dr.ic_apparie([0] * 30, [1] * 30) == (1.0, 1.0)
    lo, hi = dr.ic_apparie([0, 1] * 20, [0, 1] * 20)
    assert (lo, hi) == (0.0, 0.0)


def faux_resultats(n_prompts, delta_vraie, delta_ctrl=0.0, dnll=0.1, doses=(0.05, 0.12, 0.2)):
    prompts = {}
    for i in range(n_prompts):
        conds = {}
        for a, dd in zip((0.5, 1.0, 2.0), doses):
            conds[f"vraie@{a:g}"] = {"taux": 0.3 + dd, "delta": dd, "delta_nll": dnll}
        conds["vraie@1"].update(delta=delta_vraie, taux=0.3 + delta_vraie)
        conds["vraie@-1"] = {"taux": 0.1, "delta": -0.1, "delta_nll": dnll, "frac_modifiees": 0.5}
        for c in dr.CONTROLES:
            for a in (1.0, 2.0):
                conds[f"{c}@{a:g}"] = {"taux": 0.3, "delta": delta_ctrl, "delta_nll": dnll}
        for c in conds.values():
            c.setdefault("frac_modifiees", 0.5)
            c.setdefault("labels", [0, 1] * 5)
        prompts[str(i)] = {"idx": i, "taux_ref": 0.3, "labels_ref": [0, 1] * 5, "conds": conds, "identite_alpha0": True}
    return {"prompts": prompts, "fit": {"modele": "toy", "dtype": "float32", "couche": 2}, "selection": "propre",
            "n_eval": 10, "t_debut": 0}


def test_verdicts():
    assert dr.verdict(faux_resultats(8, 0.3))["verdict"] == "CONFIRMÉ"
    assert dr.verdict(faux_resultats(8, 0.0, doses=(0.0, 0.0, 0.0)))["verdict"] == "RÉFUTÉ"
    assert dr.verdict(faux_resultats(3, 0.3))["verdict"] == "INDÉTERMINÉ"
    # effet réel mais contrôles qui bougent autant : D ≈ 0 → réfuté (pas spécifique à la direction)
    assert dr.verdict(faux_resultats(8, 0.3, delta_ctrl=0.3))["verdict"] == "RÉFUTÉ"
    # texte dégradé : aucun prompt valide → indéterminé
    assert dr.verdict(faux_resultats(8, 0.3, dnll=2.0))["verdict"] == "INDÉTERMINÉ"
    # efficace, mais un contrôle déplace beaucoup (D passe, spécificité non) → partiel
    v = dr.verdict(faux_resultats(8, 0.5, delta_ctrl=0.2))
    assert v["P_A"] and not v["P_D"] and v["verdict"] == "PARTIEL"
    assert "VERDICT : CONFIRMÉ" in dr.rapport(faux_resultats(8, 0.3))


def test_passage_complet_fit_eval_report(eng, tmp_path):
    args = Namespace(out_dir=str(tmp_path), n_fit=24, n_eval=8, n_steps=6, t_debut=0, prompts=[0, 1, 2],
                     selection="forcee", min_rare=2, model="toy", dtype="float32", layer=COUCHE)
    info = dr.phase_fit(eng, args, COUCHE)
    assert info["couche"] == COUCHE and len(info["prompts"]) == 3
    info2, donnees = dr.charger(tmp_path)
    assert set(donnees.files) == {f"{k}{p}" for p in (0, 1, 2) for k in ("R", "tok", "nll")}
    res = dr.phase_eval(eng, args, info2, donnees)
    assert res["prompts_selectionnes"] == [0, 1, 2]
    for p in res["prompts"].values():
        assert p["identite_alpha0"]                                  # α = 0 : identité
        assert len(p["labels_ref"]) == 8
        assert set(p["conds"]) == {f"vraie@{a:g}" for a in dr.ALPHAS_VRAI} | {
            f"{c}@{a:g}" for c in dr.CONTROLES for a in dr.ALPHAS_CTRL}
        assert all(math.isfinite(c["delta_nll"]) and 0 <= c["frac_modifiees"] <= 1 for c in p["conds"].values())
    assert any(c["frac_modifiees"] > 0 for p in res["prompts"].values() for c in p["conds"].values())
    txt = dr.rapport(json.loads((tmp_path / "eval.json").read_text()))
    assert "ATTENTION : prompts forcés" in txt and "VERDICT" in txt
    # reprise : un second appel ne recalcule rien (même résultat)
    res2 = dr.phase_eval(eng, args, info2, donnees)
    assert res2["prompts"] == res["prompts"]
