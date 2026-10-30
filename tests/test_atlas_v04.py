# Tests des ajouts de l'atlas v0.4 : cartes de l'espace caché (stockage, requêtes, signatures, accord entre cartes,
# sérialisation JSON), comparabilité à trois états, provenance complète, construction de bout en bout sur le moteur
# v0.2.1 (petit Qwen3 aléatoire, CPU ; seul le tokenizer de Qwen/Qwen3-0.6B est téléchargé).

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas import compat  # noqa: E402
from atlas.bifurcation_atlas import (  # noqa: E402
    CHAMPS_SIGNATURE,
    BifurcationAtlas,
    comparer_provenances,
    p_hypergeometrique,
)
from trajectory_bifurcation_core import BifurcationResult  # noqa: E402


def fiche(prompt, multi, verdict=None, ecart=0.2, k=4):
    f = {"prompt": prompt, "multimodal": np.bool_(multi), "dip": np.float32(0.05), "p": 0.001 if multi else 0.6,
         "suites_distinctes": 1.0, "entropie_moyenne": np.float64(1.2)}
    if multi:
        f["bifurcation"] = {"k": k, "gain_max": 0.3, "k_gain_max": 5, "ratio_entropie": float("nan"),
                            "purete": [0.5], "hasard": [0.5]}
        f["valeur_ajoutee"] = ({"verdict": verdict} if verdict.startswith("NON TESTABLE") else
                               {"verdict": verdict, "ecart": ecart, "ecart_ci": (ecart - 0.1, ecart + 0.1),
                                "auc_etats": 0.8, "auc_jetons": 0.6})
    return f


def carte(fiches, couche=17, rep="rejouee"):
    m = sum(bool(f["multimodal"]) for f in fiches)
    return {"couche": couche, "t_valeur": 8, "n": 192, "n_steps": 32, "npc": 10, "seed_base": 1, "seed_analyse": 2,
            "representation": rep,
            "regimes": {"pi": m / len(fiches), "pi_ci": (0.0, 1.0), "n_multi": m, "nul": 0.0, "puissance": 1.0,
                        "verdict": "MIXTE"},
            "fiches": fiches}


PROV = {"protocole": "sgz", "modele": "Qwen/Qwen3-1.7B-Base", "sampler": "gumbel", "temperature": 0.6, "top_p": 1.0,
        "chat": False, "forbid_special": True, "q_tsallis": 1.45, "dtype": "float16", "max_new_tokens": 32}


def test_comparabilite_trois_etats():
    cles = ("modele", "temperature", "dtype")
    a = {"modele": "m", "temperature": 0.6, "dtype": "float16"}
    assert comparer_provenances(a, dict(a), cles)["comparable"] is True
    r = comparer_provenances(a, dict(a, dtype="bfloat16"), cles)
    assert r["comparable"] is False and r["differences"] == ["dtype"]
    r = comparer_provenances(a, {"modele": "m", "temperature": 0.6}, cles)       # provenance v0.3 : pas de dtype
    assert r["comparable"] is None and r["inconnus"] == ["dtype"]
    assert comparer_provenances(a, None, cles)["comparable"] is None
    assert comparer_provenances(a, dict(a, modele="x"), cles, ignorer=("modele",))["comparable"] is True


def test_reflexion_on_off_non_comparable():
    """Défaut v0.3 corrigé : deux entrées qui ne diffèrent que par la réflexion étaient « comparables »."""
    e = lambda: BifurcationResult(prompt="p", n_trajectories=8, residual_dist_over_time={},
                                  js_over_time=np.array([0.0, 0.1, 0.3]), bifurcation_times={"js": 2, "residual": None},
                                  patch_results=[], summary={})
    base = {k: PROV.get(k) for k in ("modele", "sampler", "temperature", "top_p", "chat", "max_new_tokens",
                                     "forbid_special", "dtype")}
    base.update(min_js_den=1e-4, min_ll_den=1e-3, topk_for_divergence=256)
    atlas = BifurcationAtlas()
    atlas.add(e(), dict(base, analyze_kwargs={"chat_template_kwargs": {"enable_thinking": True}}))
    atlas.add(e(), dict(base, analyze_kwargs={"chat_template_kwargs": {"enable_thinking": False}}))
    atlas.add(e(), dict(base, analyze_kwargs={"chat_template_kwargs": {"enable_thinking": True}}))
    c = atlas.compare_entries(0, 1)
    assert c["comparable"] is False and c["differences"] == ["analyze_kwargs"]
    assert atlas.compare_entries(0, 2)["comparable"] is True


def test_p_hypergeometrique():
    # 4 prompts, 2 multimodaux dans chaque carte : P(au moins 1 partagé) = 5/6, P(2) = 1/6
    assert p_hypergeometrique(4, 2, 2, 1) == pytest.approx(5 / 6)
    assert p_hypergeometrique(4, 2, 2, 2) == pytest.approx(1 / 6)
    assert p_hypergeometrique(4, 2, 2, 0) == 1.0
    # B2 après coup : 12 et 9 prompts multimodaux sur 64, 5 partagés → p ≈ 0,009
    assert p_hypergeometrique(64, 12, 9, 5) == pytest.approx(0.008566, abs=1e-5)


def test_carte_requetes_et_signature():
    fi = [fiche("a", True, "VALEUR AJOUTÉE", ecart=0.3), fiche("b", False),
          fiche("c", True, "NON TESTABLE (une seule classe ou trop peu d'exemples)", k=None),
          fiche("d", True, "AUCUNE", ecart=0.0)]
    atlas = BifurcationAtlas()
    i = atlas.add_carte(carte(fi), PROV)
    assert atlas.cartes[i]["nom"] == "Qwen/Qwen3-1.7B-Base" and atlas.metadata["n_cartes"] == 1
    json.dumps(atlas.cartes)                                 # types numpy et tuples convertis
    assert [f["prompt"] for _, f in atlas.fiches(multimodal=True)] == ["a", "c", "d"]
    assert [f["prompt"] for _, f in atlas.fiches(verdict_valeur="VALEUR AJOUTÉE")] == ["a"]
    sa, sb, sc = (atlas.signature(i, p) for p in "abc")
    assert sa["multimodal"] == 1.0 and sa["valeur_ajoutee"] == 1.0 and sa["valeur_ecart_bas"] == pytest.approx(0.2)
    assert sb["multimodal"] == 0.0 and all(math.isnan(sb[k]) for k in ("prefixe_k", "valeur_ecart", "valeur_ajoutee"))
    assert math.isnan(sc["valeur_ajoutee"]) and math.isnan(sc["prefixe_k"])         # non testée ≠ « aucune »
    assert atlas.signature(i, "d")["valeur_ajoutee"] == 0.0
    with pytest.raises(KeyError):
        atlas.signature(i, "z")
    prompts, X, noms = atlas.matrice_signatures(i)
    assert prompts == list("abcd") and X.shape == (4, len(CHAMPS_SIGNATURE)) and noms == CHAMPS_SIGNATURE
    r = atlas.resume_carte(i)
    assert r["n_multimodaux"] == 3 and r["prefixe_k_trouve"] == 2 and r["couche"] == 17
    assert r["valeur_ajoutee"] == {"VALEUR AJOUTÉE": 1, "AUCUNE": 1,
                                   "NON TESTABLE (une seule classe ou trop peu d'exemples)": 1}
    with pytest.raises(ValueError):
        atlas.add_carte({"fiches": []})


def test_accord_cartes():
    atlas = BifurcationAtlas()
    a = atlas.add_carte(carte([fiche("p1", True, "AUCUNE"), fiche("p2", True, "AUCUNE"), fiche("p3", False),
                               fiche("p4", False), fiche("seul_a", True, "AUCUNE")]), PROV, "qwen")
    b = atlas.add_carte(carte([fiche("p1", True, "AUCUNE"), fiche("p2", False), fiche("p3", True, "AUCUNE"),
                               fiche("p4", False)], couche=23),
                        dict(PROV, modele="HuggingFaceTB/SmolLM3-3B-Base"), "smol")
    r = atlas.accord_cartes(a, b)
    assert (r["n_communs"], r["multi_a"], r["multi_b"], r["partages"]) == (4, 2, 2, 1)
    assert r["attendu"] == pytest.approx(1.0) and r["p_hypergeometrique"] == pytest.approx(5 / 6)
    assert r["prompts_partages"] == ["p1"] and r["jaccard"] == pytest.approx(1 / 3)
    assert r["comparabilite"]["comparable"] is True           # modèle et couche ignorés entre modèles
    c = atlas.add_carte(carte([fiche("p1", True, "AUCUNE")], rep="generation"), PROV)
    assert atlas.accord_cartes(a, c)["comparabilite"]["differences"] == ["representation"]


def test_sauvegarde_cartes_et_atlas_v03(tmp_path):
    atlas = BifurcationAtlas("t")
    atlas.add_carte(carte([fiche("a", True, "VALEUR AJOUTÉE"), fiche("b", False)]), PROV, "qwen")
    atlas.save(tmp_path / "x")
    assert (tmp_path / "x" / "atlas_cartes.json").exists()
    lu = BifurcationAtlas.load(tmp_path / "x")
    meme = lambda x, y: json.dumps(x) == json.dumps(y)          # NaN ≠ NaN : comparaison du JSON
    assert meme(lu.cartes, atlas.cartes) and lu.metadata["n_cartes"] == 1
    assert meme(lu.signature(0, "a"), atlas.signature(0, "a"))
    s = lu.summary()
    assert s["n_entries"] == 0 and s["n_cartes"] == 1 and s["cartes"][0]["n_multimodaux"] == 1
    # Atlas v0.3 : pas de fichier de cartes
    (tmp_path / "x" / "atlas_cartes.json").unlink()
    assert BifurcationAtlas.load(tmp_path / "x").cartes == []
    # Export / import d'une carte seule (processus séparés)
    atlas.exporter_carte(0, tmp_path / "c.json")
    autre = BifurcationAtlas()
    assert autre.importer_carte(tmp_path / "c.json") == 0 and meme(autre.cartes[0], atlas.cartes[0])


def test_print_summary_cartes_puis_historique(capsys):
    atlas = BifurcationAtlas("t")
    atlas.add_carte(carte([fiche("a", True, "AUCUNE"), fiche("b", False)]), PROV, "qwen")
    atlas.add_carte(carte([fiche("a", True, "AUCUNE"), fiche("b", True, "AUCUNE")], couche=23),
                    dict(PROV, modele="smol"), "smol")
    atlas.print_summary()
    out = capsys.readouterr().out
    assert out.index("Cartes de l'espace caché") < out.index("Entrées historiques")
    assert "accord qwen / smol : 1 prompts multimodaux partagés sur 2" in out


def test_verifier_moteur_requiert_cartographier_v021():
    class Ancien(compat.TrajectoryBifurcationEngine):
        def cartographier(self, prompts, n=192, n_steps=32, layer=None, t_valeur=None, seed_base=None, npc=10,
                          seed_analyse=0):
            pass
    with pytest.raises(RuntimeError, match=r"cartographier\(rejouer=\)"):
        compat.verifier_moteur(Ancien)
    compat.verifier_moteur()


@pytest.fixture(scope="module")
def jouet():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("sklearn")
    pytest.importorskip("diptest")
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
    return model, tok


def test_cartographier_de_bout_en_bout(jouet):
    model, tok = jouet
    atlas = BifurcationAtlas()
    prompts = ["La ville de Lyon est située au confluent", "The city of Boston is known for",
               "Il était une fois", "Pour préparer une pâte"]
    kw = dict(model=model, tokenizer=tok, device="cpu", model_name="toy-Base", max_new_tokens=10,
              layers_to_track=[0, 2, 3], batch_size=8)
    i = atlas.cartographier("sgz", prompts, nom="jouet", moteur_kwargs=kw, n=24, n_steps=10, t_valeur=3, npc=4,
                            rejouer=True)
    par = atlas.parametres_carte(i)
    assert par["protocole"] == "sgz" and par["dtype"] == "float32" and par["device"] == "cpu"
    assert par["forbid_special"] is True and par["q_tsallis"] == 1.45 and par["representation"] == "rejouee"
    assert par["couche"] == 1 and par["n"] == 24 and par["core_version"] == "0.2.1"   # ⌊2·4/3⌋ − 1
    r = atlas.resume_carte(i)
    assert r["n_prompts"] == 4 and r["regimes"]["verdict"]
    _, X, _ = atlas.matrice_signatures(i)
    assert X.shape == (4, len(CHAMPS_SIGNATURE)) and set(X[:, 0]) <= {0.0, 1.0}
    with pytest.raises(ValueError):
        atlas.cartographier("sgz", prompts, moteur_kwargs=kw, n=24, inconnu=1)
