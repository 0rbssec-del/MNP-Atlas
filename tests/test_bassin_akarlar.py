# Test du montage de experiments/bassin_akarlar.py sur un petit Qwen3 aléatoire (CPU) : nombres aléatoires communs,
# auto-patch identique à la base, patch persistant qui change la suite, analyse et verdicts.
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def eng():
    from trajectory_bifurcation_core import TrajectoryBifurcationEngine
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # pas de réseau ni de cache
        pytest.skip(f"Tokenizer Qwen3 indisponible : {e}")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    model = transformers.Qwen3ForCausalLM(cfg).float()
    with torch.no_grad():
        model.lm_head.weight.mul_(3.0)
    return TrajectoryBifurcationEngine(model=model, tokenizer=tok, model_name="toy", device="cpu", sampler="gumbel",
                                      temperature=0.7, top_p=1.0, chat=False, forbid_special=False, seed=1,
                                      max_new_tokens=15, layers_to_track=[0, 2, 3], batch_size=16)


PROMPTS = [{"idx": 1, "prompt": "Il était une fois", "ground_truth": ["a"], "wrong_indicators": ["e"]},
           {"idx": 2, "prompt": "Une histoire courte", "ground_truth": ["i"], "wrong_indicators": ["o"]}]


def bassin_parite(ba, *args, **kw):
    """Bassin dont l'étiquette est la parité de la somme des jetons. Sur un modèle aléatoire, 15 jetons tirés dans
    tout le vocabulaire contiennent presque toujours « a » : le classifieur par sous-chaînes classerait tout le
    réservoir CORRECT, sans paire discordante. La parité donne des deux classes et dépend de toute la suite, donc le
    patch peut la changer."""
    class BassinParite(ba.Bassin):
        def etiquettes(self, ids, p):
            lab = "CORRECT" if sum(int(i) for i in ids) % 2 == 0 else "HALLUCINATION"
            return lab, lab, ""
    return BassinParite(*args, **kw)


def test_etiquettes_par_le_classifieur(eng):
    import bassin_akarlar as ba
    b = ba.Bassin(eng, PROMPTS)
    p = {"ground_truth": ["Paris"], "wrong_indicators": ["Lyon"]}
    for mot, lab in (("Paris", "CORRECT"), ("Lyon", "HALLUCINATION"), ("Nice", "OTHER")):
        ids = eng.tokenizer(" " + mot)["input_ids"]
        assert b.etiquettes(ids, p)[:2] == (lab, lab)


def test_montage_et_analyse(eng):
    import bassin_akarlar as ba
    prompts = PROMPTS
    b = bassin_parite(ba, eng, prompts, n_pool=16, n_pairs=2, n_seeds=4, layers=[0, 2, 3], l_soi=2)
    meta = b.run()
    assert all(m["paires"] for m in meta)
    conds = {r["cond"] for r in b.rows}
    assert conds == {"base", "echange", "soi", "L0", "L2", "L3"}
    res = ba.analyser(b.rows, b.layers, b.l_soi)
    assert res["soi_identique"] >= 0.98                      # témoin négatif : auto-patch = base
    assert set(res["ci"]) >= {"A", "C->H_moy", "H->C_moy"} and res["verdict"]
    txt = ba.rapport(res, b.layers)
    assert "[A] asymétrie" in txt and res["verdict"] in txt
    # nombres aléatoires communs : même (paire, direction, graine) → mêmes lignes de bruit, donc base = base relancée
    base1 = [(r["idx"], r["pair"], r["dir"], r["seed"], r["t1"]) for r in b.rows if r["cond"] == "base"]
    b2 = bassin_parite(ba, eng, prompts, n_pool=16, n_pairs=2, n_seeds=4, layers=[0], l_soi=0)
    b2.run()
    base2 = [(r["idx"], r["pair"], r["dir"], r["seed"], r["t1"]) for r in b2.rows if r["cond"] == "base"]
    assert base1 == base2


def test_patch_persistant_change_la_suite(eng):
    import bassin_akarlar as ba
    pid = list(eng.tokenizer("Il était une fois")["input_ids"])
    b = ba.Bassin(eng, [], layers=[0, 2, 3], l_soi=2)
    st = b.etats([pid + [100], pid + [200]])
    rows = list(range(8))
    base = b.lots([pid + [100]] * 8, rows, 6, 5, 1)
    soi = b.lots([pid + [100]] * 8, rows, 6, 5, 1, 2, st[2][[0] * 8])
    autre = b.lots([pid + [100]] * 8, rows, 6, 5, 1, 0, st[0][[1] * 8])
    assert (base == soi).all()                                # même état → même suite (même bruit)
    assert (base != autre).any()                              # état d'un autre jeton à L0 → suite changée
