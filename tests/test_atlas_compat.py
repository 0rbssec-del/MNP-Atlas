# Tests de la couche atlas (atlas/compat.py) sur le moteur v0.2, avec un petit
# Qwen3 à poids aléatoires (CPU). Seul le tokenizer de Qwen/Qwen3-0.6B est
# téléchargé.

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from atlas import compat  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def jouet():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # pas de réseau ni de cache
        pytest.skip(f"Tokenizer Qwen3 indisponible : {e}")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512,
    )
    model = transformers.Qwen3ForCausalLM(cfg).float()
    with torch.no_grad():
        model.lm_head.weight.mul_(20.0)
    return model, tok


def moteur(jouet, protocole, **kw):
    model, tok = jouet
    kw.setdefault("max_new_tokens", 12)
    kw.setdefault("layers_to_track", [0, 2, 3])
    return compat.construire_moteur(
        protocole, model=model, tokenizer=tok, device="cpu", model_name="toy", **kw
    )


@pytest.mark.parametrize("protocole", list(compat.PROTOCOLES))
def test_protocole_applique(jouet, protocole):
    eng = moteur(jouet, protocole)
    for k, v in compat.PROTOCOLES[protocole]["moteur"].items():
        attr = {"q_tsallis": "q"}.get(k, k)
        assert getattr(eng, attr) == v, k
    seuils = compat.PROTOCOLES[protocole]["seuils_recovery"]
    if seuils is not None:
        assert (eng.MIN_JS_DEN, eng.MIN_LL_DEN) == seuils
    assert eng.protocole_atlas == protocole


def test_reglage_contraire_refuse(jouet):
    with pytest.raises(ValueError, match="temperature"):
        moteur(jouet, "v015", temperature=0.6)
    with pytest.raises(ValueError, match="Protocole inconnu"):
        moteur(jouet, "v0.2")


def test_moteur_sans_api_v02_refuse():
    class Ancien:
        def __init__(self, model_name="x", temperature=0.8):
            pass

        def _patch_hook(self):
            pass

    with pytest.raises(RuntimeError, match="sampler.*chat.*_capture_hooks"):
        compat.verifier_moteur(Ancien)


def tirages_v015(model, prompt_ids, n, T, top_p, graine, eos):
    """Référence indépendante de l'échantillonneur v0.1.5 : KV-cache,
    softmax(z / T), top-p par tri, torch.multinomial à graine, arrêt sur EOS."""
    gen = torch.Generator().manual_seed(graine)
    past, cur, out = None, prompt_ids, []
    with torch.no_grad():
        for _ in range(n):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            p = F.softmax(o.logits[:, -1].float() / T, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(p, descending=True)
                m = torch.cumsum(sp, -1) > top_p
                m[..., 1:] = m[..., :-1].clone()
                m[..., 0] = False
                sp = sp.masked_fill(m, 0.0)
                sp = sp / (sp.sum(-1, keepdim=True) + 1e-12)
                nxt = si.gather(-1, torch.multinomial(sp, 1, generator=gen))
            else:
                nxt = torch.multinomial(p, 1, generator=gen)
            out.append(int(nxt))
            if out[-1] in eos:
                break
            cur = nxt.view(1, 1)
    return out


@pytest.mark.parametrize("protocole", ["v015", "akarlar"])
def test_tirages_identiques_a_v015(jouet, protocole):
    """Les protocoles figés refont les tirages de l'échantillonneur v0.1.5."""
    eng = moteur(jouet, protocole)
    ids = eng.build_prompt_ids("Raconte une histoire courte.")
    for graine in (0, 1, 7):
        tr = compat.generer_trajectoire(eng, ids, graine)
        ref = tirages_v015(eng.model, ids, eng.max_new_tokens, eng.temperature,
                           eng.top_p, graine, eng.eos_token_ids)
        assert tr.generated_ids == ref


def test_gabarit_selon_protocole(jouet):
    v015, sgz = moteur(jouet, "v015"), moteur(jouet, "sgz")
    brut = list(sgz.tokenizer.encode("Bonjour"))
    assert sgz.build_prompt_ids("Bonjour")[0].tolist() == brut
    assert len(v015.build_prompt_ids("Bonjour")[0]) > len(brut)


def test_acces_internes(jouet):
    eng = moteur(jouet, "sgz")
    p = torch.softmax(torch.randn(2, 50, dtype=torch.float64), -1)
    assert torch.allclose(compat.js_exacte(p, p), torch.zeros(2, dtype=torch.float64))
    ids = eng.build_prompt_ids("Bonjour")
    with compat.capture_hooks(eng, [1]) as cap:
        eng.model(input_ids=ids)
    vec = cap[1] * 0 + 3.0
    with compat.patch_hook(eng, 1, ids.shape[1] - 1, vec), compat.capture_hooks(eng, [1]) as cap2:
        eng.model(input_ids=ids)
    assert torch.allclose(cap2[1], vec)


def test_paires_du_type_json_avant_v02():
    paires = [
        {"kind": "informative", "last_token_same": True},
        {"kind": "informative", "last_token_same": False},
        {"kind": "informative", "detail": "informative_token", "last_token_same": False},
        {"kind": "positive_control", "detail": "positive_control"},
    ]
    assert len(compat.paires_du_type(paires, "informative_context")) == 1
    assert len(compat.paires_du_type(paires, "informative_token")) == 2
    assert len(compat.paires_du_type(paires, "positive_control")) == 1


@pytest.mark.parametrize("protocole", ["v015", "sgz"])
def test_multi_prompt_run_one(jouet, protocole):
    import multi_prompt

    eng = moteur(jouet, protocole, max_new_tokens=10)
    args = SimpleNamespace(n=6, seed=0, n_pairs=2, layers=[0, 2], cont=4,
                           horizon=4, detail=protocole == "sgz")
    thinking = None if protocole == "sgz" else True
    out = multi_prompt.run_one(eng, "Raconte une histoire courte.", thinking, args,
                               np.random.default_rng(0))
    assert out["thinking"] is thinking
    assert out["corr_js_divergence_tokens"] is None or -1 <= out["corr_js_divergence_tokens"] <= 1
    if protocole == "sgz":
        assert out["min_len"] == 10          # jetons spéciaux interdits : longueur fixe
    for paires in out["layers"].values():
        for p in paires:
            assert p["detail"] is not None
            if p["kind"] == "negative_control":
                assert p["overlap"] == 1.0
