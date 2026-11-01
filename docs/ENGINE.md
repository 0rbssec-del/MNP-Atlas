# Moteur v0.2.1 : `trajectory_bifurcation_core.py`

Version : Core v0.2.1 (branche `ccr-0f9a5c2e-elasx1`, commit `0af62ba`). La v0.2.0 a été livrée au commit `d900d27` ;
la v0.2.1 ajoute `cartographier(rejouer=…)` (voir § 3). Le code atlas ne modifie pas le moteur (voir `docs/ATLAS.md`).
1 840 lignes ; tests : `tests/test_trajectory_bifurcation_core.py`, `tests/test_v02.py`.

## 1. Rôle

Le moteur génère, pour un prompt fixe, un ensemble de trajectoires échantillonnées d'un modèle causal Hugging Face,
mesure où et comment elles divergent (distributions du jeton suivant, flux résiduel), et teste par *activation patching*
si l'état caché d'une trajectoire porte l'issue d'une autre. La v0.2 l'aligne sur le protocole des notebooks SGZ
(A1c–B2) et ajoute des analyses de régimes.

## 2. Architecture

| Bloc | Contenu |
|---|---|
| Structures | `Trajectory`, `Continuation`, `Ensemble` (lots), `PatchResult`, `BifurcationResult` |
| Fonctions statistiques | entropies Shannon / Tsallis, bruit de Gumbel, `analyser_regimes` (B2), `bifurcation_prefixe`, `valeur_ajoutee_sur_jetons` |
| Moteur : construction | `__init__`, `sgz()` (réglages SGZ) |
| Hooks | capture de la sortie brute des couches, injection (patch) |
| Échantillonnage | Gumbel à graine par (ligne, pas), multinomial v0.1.x, masque des jetons spéciaux, top-p |
| Génération par lots | `generer_lot`, `generer_ensemble`, `rejouer_moyennes`, `ensemble_choc` (A1) |
| API v0.1.x | `_decode_loop` (KV-cache), `build_prompt_ids`, `generate_ensemble` |
| Divergences | JS top-k, classement des paires, `compute_divergences`, `detect_bifurcation` |
| Patching | `continue_from_prefix`, `forced_patch_effect`, `choose_patch_time`, `activation_patch_test`, `reversibility_analysis`, `summarize_patches` |
| Pipelines | `analyze_prompt`, `cartographier` |

## 3. Points techniques

**Échantillonneur.** Par défaut `sampler="gumbel"` : jeton = argmax(z / T + G), où G est un bruit de Gumbel dont la
graine vaut `seed + 100003·ligne + pas`. Deux branches qui partagent un préfixe reçoivent le même bruit à chaque position
(nombres aléatoires communs) : un patch nul laisse la continuation identique. `sampler="multinomial"` reproduit
v0.1.x (`torch.multinomial` + générateur à graine). Avec Gumbel, les jetons spéciaux sont interdits (longueur fixe).

**Défauts v0.2.** T = 0,6, top-p = 1, prompt brut pour les modèles « -Base » (gabarit de chat sinon, `chat=` pour
forcer), fp16 sur GPU sans bf16 natif, float32 sur CPU, entropie de Tsallis q = 1,45 en plus de Shannon (loi du modèle à
T = 1).

**Convention des états.** `hidden_states[l][t]` est la sortie brute de la couche l (avant la norme finale) à la dernière
position, quand le modèle calcule la loi du jeton t. Couche l du moteur = `hidden_states[l + 1]` de Hugging Face. Les
hooks capturent et injectent au même point, ce qui évite la double norme de la dernière couche.

**Patch persistant.** `_patch_hook` remplace la sortie de la couche à une position lors du premier passage ; avec le
KV-cache, les clés/valeurs des couches suivantes, calculées depuis l'état patché, restent dans le cache. Les positions
suivantes voient donc le patch (contrairement à un recalcul sans cache qui l'efface).

**Classement des paires** (préfixe de la cible jusqu'à t) : `negative_control` (préfixes identiques, patch nul),
`positive_control` (seul le dernier jeton diffère : recovery = 1 par construction), `informative_context` (dernier jeton
identique, le patch ne transmet que du contexte), `informative_token` (dernier jeton différent : le patch transmet aussi
son identité).

**Mesures de recovery.** `recovery` : 1 − JS(patché, source) / JS(cible, source) au premier pas (top-k).
`forced_patch_effect` force la continuation de la source sur `horizon` jetons et calcule, sur distributions exactes,
`recovery_multistep` (rapport des sommes de JS) et `recovery_loglik` (part de log-vraisemblance récupérée). Rapports à
None si le dénominateur est sous `MIN_JS_DEN = 0,05` nat ou `MIN_LL_DEN = 0,5` nat.

**Divergences.** `compute_divergences` donne JS(t), distances L2 et cosinus, la part des paires au même préfixe et
`corr_js_divergence_jetons` : une corrélation ≥ 0,9 signifie que la « bifurcation » JS suit la divergence des jetons, pas
une dynamique interne. `detect_bifurcation` : premier t au-dessus de max(ratio × ligne de base, plancher absolu).

**Régimes (B2).** `analyser_regimes` teste la bimodalité par moitiés indépendantes (2-moyennes sur A, dip de Hartigan
sur B), calibre par substitut gaussien, mesure la puissance par mélanges de prompts, corrige par Benjamini-Hochberg ;
verdict BASSINS / MIXTE / CONTINUUM / NON INTERPRÉTABLE. `valeur_ajoutee_sur_jetons` compare l'AUC hors échantillon
d'une sonde sur l'état caché à celle d'un lecteur de jetons du même préfixe.

**Cartes (v0.2.1).** `cartographier(prompts, n, n_steps, layer, t_valeur, seed_base, npc, seed_analyse, rejouer)`
génère l'ensemble par lots, calcule les régimes de B2 sur la représentation R_h (moyenne des états de la couche sur la
suite) et, pour chaque prompt multimodal, la bifurcation par préfixe et la valeur ajoutée de l'état au pas `t_valeur`.
Nouveautés v0.2.1 : `rejouer=True` recalcule R_h par une passe complète (`rejouer_moyennes`, exactement la représentation
de B2) au lieu de la moyenne accumulée pendant la génération ; `t_valeur` est vérifié (1 ≤ t_valeur < n_steps − 1) ; la
carte renvoie les réglages qui fixent le résultat (n, n_steps, npc, graines, représentation).

## 4. Utilisation

```python
from trajectory_bifurcation_core import TrajectoryBifurcationEngine
eng = TrajectoryBifurcationEngine.sgz("Qwen/Qwen3-1.7B-Base", max_new_tokens=32, seed=20261029)
ens = eng.generer_ensemble(prompts, n=192, n_steps=32, layers=[17], keep_steps=True, mean_layers=[17])
carte = eng.cartographier(prompts, n=192, n_steps=32, layer=17, t_valeur=8, rejouer=True)
res = eng.analyze_prompt("The city of Boston is known for", n_trajectories=12, require_bifurcation=False)
```

## 5. Limites connues

- La sortie de la dernière couche après la norme finale n'est pas capturée par les hooks.
- Le patching traite une séquence à la fois.
- `analyze_prompt` ne réserve pas de quota de paires « contexte » (`reversibility_analysis(detail=True)` le fait).

## 6. Code complet

```python
# trajectory_bifurcation_core.py
# Core v0.2.1 – Trajectory Bifurcation Engine
# Fonctionnalités :
#   - Génération d'ensembles de trajectoires à prompt fixe, par lots (remplissage
#     à gauche, KV-cache), avec deux échantillonneurs :
#       « gumbel » (défaut) : bruit de Gumbel à graine explicite par ligne et par
#         position (nombres aléatoires communs, protocole SGZ A1c–B2) ;
#       « multinomial » : échantillonneur de v0.1.x (torch.multinomial + graine).
#   - Entropies de Shannon et de Tsallis (q = 1,45) de la loi du modèle (T = 1).
#   - Divergences : Jensen-Shannon (top-k + masse résiduelle) + distances
#     résiduelles (L2 et cosinus) ; part des paires au même préfixe.
#   - Régimes (test de B2) : bimodalité par moitiés indépendantes (2-moyennes sur
#     A, test du dip sur B), calibration par substitut gaussien, puissance par
#     mélanges de prompts, Benjamini-Hochberg.
#   - Bifurcation par préfixe (B2) : premier k où le préfixe prédit le régime.
#   - Valeur ajoutée sur les jetons : l'état caché à t prédit-il l'issue mieux
#     qu'un lecteur de jetons du même préfixe ? (AUC hors échantillon).
#   - Choc (protocole A1) : jeton de rang imposé à t0, témoin et choc couplés.
#   - Test de réversibilité par activation patching (hooks), paires classées en
#     témoins (négatif, positif), « contexte » (dernier jeton identique) et
#     « jeton » (dernier jeton différent).
#
# Changements v0.1.5 → v0.2.0 :
#   - Échantillonneur « gumbel » par défaut, identique à celui des notebooks SGZ
#     (même graine → mêmes jetons ; les témoins négatifs ne dépendent plus d'un
#     arrondi dans torch.multinomial). « multinomial » reste disponible.
#   - Défauts alignés sur le protocole SGZ : T = 0,6, pas de top-p, prompt brut
#     pour les modèles de base (gabarit de chat seulement s'il est demandé ou si
#     le modèle n'est pas « -Base »), fp16 sur GPU sans bf16 natif (T4).
#   - Génération par lots (generer_ensemble), états cachés capturés par lot.
#   - Paires « informative » scindées en « contexte » / « jeton » : le recovery
#     des paires « jeton » mesure surtout l'identité du dernier jeton.
#   - Recovery : None si le dénominateur est trop petit (instabilité, témoin nul
#     à |recovery| > 100 dans v0.1.5).
#   - Corrélation JS / divergence des jetons rapportée (critère C1 de v0.1.5) :
#     une « bifurcation » JS n'est pas, à elle seule, une dynamique interne.

from __future__ import annotations

import contextlib
import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

TopK = Tuple[np.ndarray, np.ndarray, float]
PAS_GRAINE = 100003                      # graine(ligne, pas) = base + 100003·ligne + pas (SGZ)

# --------------------------------------------------------------------------
# Structures de données
# --------------------------------------------------------------------------

@dataclass
class Trajectory:
    token_ids: List[int]
    # Complétion seule (sans le prompt)
    text: str
    prompt_len: int = 0
    seed: Optional[int] = None
    entropy_shannon: List[float] = field(default_factory=list)
    # Sortie brute de la couche l (avant norme finale), dernière position :
    # hidden_states[l][t] = état qui produit la loi du jeton généré t
    hidden_states: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    # (top_indices, top_probs, remaining_mass)
    topk_distributions: List[TopK] = field(default_factory=list)
    entropy_tsallis: List[float] = field(default_factory=list)

    @property
    def generated_ids(self) -> List[int]:
        return self.token_ids[self.prompt_len:]

@dataclass
class Continuation:
    token_ids: List[int]
    text: str
    first_distribution: TopK

@dataclass
class Ensemble:
    """Ensemble de trajectoires générées par lots (longueur fixe).

    tokens[i, k] : jeton généré k de la ligne i ; entropies[i, k] : loi de ce
    jeton (T = 1) ; hidden_steps[l][i, k] : état de la couche l à la dernière
    position quand la loi du jeton k est calculée (lecture du jeton k − 1, ou du
    dernier jeton du prompt pour k = 0) ; hidden_mean[l][i] : moyenne des états
    après lecture des jetons générés 0 … T − 1 (représentation R_h de B2).
    """
    prompt_ids: List[List[int]]
    prompt_of: np.ndarray
    row_ids: np.ndarray
    tokens: np.ndarray
    entropy_shannon: np.ndarray
    entropy_tsallis: np.ndarray
    hidden_steps: Dict[int, np.ndarray] = field(default_factory=dict)
    hidden_mean: Dict[int, np.ndarray] = field(default_factory=dict)
    topk: Optional[List[List[TopK]]] = None
    # Pas dont les états sont gardés dans hidden_steps (None = tous)
    kept_steps: Optional[List[int]] = None
    # Longueur jusqu'à la première fin de texte incluse (= T si les jetons spéciaux sont interdits)
    lengths: Optional[np.ndarray] = None
    seed_base: int = 0
    step0: int = 0
    temperature: float = 0.6
    q: float = 1.45

    def lignes_du_prompt(self, p: int) -> np.ndarray:
        return np.nonzero(self.prompt_of == p)[0]

    def textes(self, tokenizer, lignes=None) -> List[str]:
        lignes = range(len(self.tokens)) if lignes is None else lignes
        return [tokenizer.decode(self.tokens[i].tolist(), skip_special_tokens=True) for i in lignes]

@dataclass
class PatchResult:
    source_traj_idx: int
    target_traj_idx: int
    layer: int
    t_patch: int
    original_continuation: str
    patched_continuation: str
    final_token_overlap: float
    success: bool
    seed: Optional[int] = None
    # "negative_control" : préfixes identiques → patch nul par construction
    # "positive_control" : seul le dernier token du préfixe diffère → contexte
    #   identique, recovery = 1 par construction
    # "informative" : les préfixes diffèrent avant la position patchée
    pair_kind: str = "informative"
    prefixes_identical: bool = False
    # Premier pas où les continuations couplées diffèrent (None = jamais)
    first_divergence_step: Optional[int] = None
    # JS au pas patché (nats) : cible/source, patché/source, patché/cible
    js_target_source: Optional[float] = None
    js_patched_source: Optional[float] = None
    js_patched_target: Optional[float] = None
    # 1 - JS(patché, source) / JS(cible, source) : 1 = se comporte comme la
    # source, 0 = aucun effet. None si cible et source sont indiscernables.
    recovery: Optional[float] = None
    # Contrôle numérique : JS entre la passe propre et la distribution
    # enregistrée pendant la génération (doit être ~0)
    js_clean_vs_recorded: Optional[float] = None
    # Indice du premier token généré qui diffère entre source et cible
    first_diff_step: Optional[int] = None
    # Dernier token du préfixe identique ? Sinon, patcher la dernière position
    # transmet aussi l'identité de ce token, pas seulement le contexte.
    last_token_same: Optional[bool] = None
    # Continuation de la source forcée sur forced_horizon tokens, distributions
    # exactes : 1 - Σ_k JS(patché_k, source_k) / Σ_k JS(cible_k, source_k)
    recovery_multistep: Optional[float] = None
    # (LL patché - LL cible) / (LL source - LL cible), LL = log-vraisemblance
    # de la continuation de la source. Continu, contrairement à recovery.
    recovery_loglik: Optional[float] = None
    recovery_per_step: Optional[List[Optional[float]]] = None
    forced_horizon: int = 0
    # v0.2 : "negative_control", "positive_control", "informative_context"
    # (dernier jeton identique, contexte antérieur différent) ou
    # "informative_token" (dernier jeton différent aussi)
    pair_detail: str = "informative_token"

@dataclass
class BifurcationResult:
    prompt: str
    n_trajectories: int
    residual_dist_over_time: Dict[int, np.ndarray]
    js_over_time: np.ndarray
    bifurcation_times: Dict[str, Optional[int]]
    patch_results: List[PatchResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    residual_cos_over_time: Dict[int, np.ndarray] = field(default_factory=dict)
    frac_pairs_same_prefix: Optional[np.ndarray] = None

# --------------------------------------------------------------------------
# Mesures (communes aux notebooks SGZ)
# --------------------------------------------------------------------------

@torch.no_grad()
def entropies_from_logits(logits: torch.Tensor, qs: Sequence[float]) -> Dict[float, torch.Tensor]:
    """Entropies du softmax complet (float64). q = 1 : Shannon (nats) ; sinon Tsallis S_q = (1 − Σ p^q)/(q − 1)."""
    logp = torch.log_softmax(logits.double(), dim=-1)
    p = logp.exp()
    out = {}
    for q in qs:
        if q == 1.0:
            out[q] = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
        else:
            out[q] = (1.0 - (p ** q).sum(-1)) / (q - 1.0)
    return out


def gumbel_noise(V: int, seeds: Sequence[int], device) -> torch.Tensor:
    """Bruit de Gumbel par ligne, graine explicite par ligne (identique aux notebooks SGZ)."""
    g = torch.Generator(device=device)
    rows = []
    for s in seeds:
        g.manual_seed(int(s))
        rows.append(torch.rand(V, generator=g, device=device, dtype=torch.float32))
    u = torch.stack(rows).clamp_(1e-10, 1 - 1e-7)
    return -torch.log(-torch.log(u))


def ci(v) -> Tuple[float, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (float("nan"), float("nan"))

# --------------------------------------------------------------------------
# Régimes, bifurcations par préfixe, valeur ajoutée (fonctions pures, CPU)
# --------------------------------------------------------------------------

def standardiser(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu, sd = ref.mean(0), ref.std(0)
    sd = sd + 1e-6 * sd.max() + 1e-12
    return (X - mu) / sd


def acp(Z: np.ndarray, k: int) -> np.ndarray:
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return Vt[:min(k, Vt.shape[0])].T


def test_bimodal(X: np.ndarray, seed: int, npc: int = 10) -> Tuple[float, float]:
    """Moitiés indépendantes (B2) : direction des 2-moyennes ajustée sur A (standardisation, ACP à npc composantes), test du
    dip de Hartigan sur la projection de B. Renvoie (dip, p)."""
    import diptest
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    A, B = X[perm[:len(X) // 2]], X[perm[len(X) // 2:]]
    Az, Bz = standardiser(A, A), standardiser(B, A)
    V = acp(Az, npc)
    PA, PB = Az @ V, Bz @ V
    if np.ptp(PA, axis=0).max() < 1e-9:
        return 0.0, 1.0
    km = KMeans(2, n_init=10, random_state=int(seed % 2 ** 31)).fit(PA)
    u = km.cluster_centers_[0] - km.cluster_centers_[1]
    u = u / (np.linalg.norm(u) + 1e-12)
    proj = PB @ u
    if np.ptp(proj) < 1e-9:
        return 0.0, 1.0
    d, p = diptest.diptest(proj.astype(np.float64))
    return float(d), float(p)


def substitut_gaussien(X: np.ndarray, rng) -> np.ndarray:
    """Même moyenne et même covariance empirique que X, loi gaussienne (unimodale)."""
    Xc = X - X.mean(0)
    return X.mean(0) + rng.standard_normal((len(X), len(X))) @ Xc / np.sqrt(len(X) - 1)


def bh(p, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, n + 1) / n
    k = int(np.nonzero(ok)[0].max()) + 1 if ok.any() else 0
    rej = np.zeros(n, bool)
    rej[o[:k]] = True
    return rej


def wilson(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(c - h), float(c + h))


def regle_regimes(pi: float, nul: float, puissance: float, seuil_bassins: float = 0.5, seuil_continuum: float = 0.1,
                  seuil_nul: float = 0.1, seuil_puissance: float = 0.8) -> str:
    """Règle de B2 : BASSINS si π ≥ 0,5 ; CONTINUUM si π ≤ 0,1 (et puissance ≥ 0,8) ; sinon MIXTE."""
    if nul > seuil_nul:
        return "NON INTERPRÉTABLE (test mal calibré sur le substitut gaussien)"
    if pi >= seuil_bassins:
        return "BASSINS"
    if pi <= seuil_continuum:
        return "CONTINUUM" if puissance >= seuil_puissance else "NON INTERPRÉTABLE (test sans puissance)"
    return "MIXTE"


def analyser_regimes(R: Sequence[np.ndarray], seed: int = 0, npc: int = 10, alpha: float = 0.05,
                     paires: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """R[p] : (n_p, d), une représentation par trajectoire du prompt p. Test par prompt, substitut gaussien, mélanges de
    paires de prompts (puissance), correction de Benjamini-Hochberg, verdict de B2."""
    P = len(R)
    rng = np.random.default_rng(seed)
    vrai = [test_bimodal(np.asarray(R[p], np.float64), seed + p, npc) for p in range(P)]
    nul = [test_bimodal(substitut_gaussien(np.asarray(R[p], np.float64), rng), seed + 1000 + p, npc) for p in range(P)]
    paires = [(2 * i, 2 * i + 1) for i in range(P // 2)] if paires is None else list(paires)
    mix = []
    for a, b in paires:
        na, nb = len(R[a]), len(R[b])
        m = min(na, nb) // 2
        ia, ib = rng.permutation(na)[:m], rng.permutation(nb)[:m]
        mix.append(test_bimodal(np.concatenate([R[a][ia], R[b][ib]]).astype(np.float64), seed + 2000 + a, npc))
    rej = bh([p for _, p in vrai], alpha)
    pi = float(rej.mean()) if P else float("nan")
    tn = float(bh([p for _, p in nul], alpha).mean()) if P else float("nan")
    pu = float(bh([p for _, p in mix], alpha).mean()) if mix else float("nan")
    return {"pi": pi, "pi_ci": wilson(int(rej.sum()), P), "n_multi": int(rej.sum()), "nul": tn, "puissance": pu,
            "verdict": regle_regimes(pi, tn, pu if mix else 1.0), "dip": [d for d, _ in vrai], "p": [p for _, p in vrai],
            "multimodal": rej.tolist()}


def etiquettes_regimes(X: np.ndarray, seed: int = 0, npc: int = 10) -> np.ndarray:
    """Étiquettes des deux régimes (2-moyennes sur l'ACP standardisée), comme B2."""
    from sklearn.cluster import KMeans
    Z = standardiser(X, X)
    return KMeans(2, n_init=10, random_state=int(seed % 2 ** 31)).fit_predict(Z @ acp(Z, npc))


def purete(cles: Sequence, lab: np.ndarray) -> float:
    grp: Dict[Any, List[int]] = {}
    for c, l in zip(cles, lab):
        grp.setdefault(c, []).append(int(l))
    return sum(max(np.bincount(v)) for v in grp.values()) / len(lab)


def bifurcation_prefixe(TK: np.ndarray, lab: np.ndarray, S: Optional[np.ndarray] = None, seed: int = 0,
                        seuil: float = 0.9, ecart: float = 0.25, n_perm: int = 20) -> Dict[str, Any]:
    """Premier k où le préfixe de k jetons prédit le régime à ≥ seuil, à ≥ écart au-dessus du hasard (étiquettes
    permutées) ; gain maximal du préfixe sur le hasard et sa position ; entropie à k* rapportée aux autres positions."""
    rng = np.random.default_rng(seed)
    acc, hasard = [], []
    for k in range(1, TK.shape[1] + 1):
        cles = [tuple(r[:k]) for r in TK]
        acc.append(purete(cles, lab))
        hasard.append(float(np.mean([purete(cles, rng.permutation(lab)) for _ in range(n_perm)])))
    acc, hasard = np.array(acc), np.array(hasard)
    ok = np.nonzero((acc >= seuil) & (acc - hasard >= ecart))[0]
    k = int(ok[0]) + 1 if ok.size else None
    gain = acc - hasard
    ratio = float("nan")
    if k is not None and S is not None:
        prof = S.mean(0)
        ratio = float(prof[k - 1] / np.median(np.delete(prof, k - 1)))
    return {"k": k, "gain_max": float(gain.max()), "k_gain_max": int(gain.argmax()) + 1, "ratio_entropie": ratio,
            "purete": acc.tolist(), "hasard": hasard.tolist()}


def sac_de_jetons(prefixes: Sequence[Sequence[int]], n_hash: int = 4096, n_derniers: int = 3) -> np.ndarray:
    """Lecteur de jetons : sac des jetons du préfixe (log(1 + compte), hachage) et identité des n derniers jetons avec
    leur position (hachage). Aucune information du modèle."""
    X = np.zeros((len(prefixes), 2 * n_hash), np.float32)
    for i, pr in enumerate(prefixes):
        for t in pr:
            X[i, int(t) % n_hash] += 1.0
        for j, t in enumerate(list(pr)[-n_derniers:][::-1]):
            X[i, n_hash + zlib.crc32(f"{j}:{int(t)}".encode()) % n_hash] = 1.0
    X[:, :n_hash] = np.log1p(X[:, :n_hash])
    return X


def _oof_logistique(X: np.ndarray, y: np.ndarray, groupes: np.ndarray, plis: int, C: float) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    s = np.zeros(len(y))
    for tr, te in GroupKFold(plis).split(X, y, groupes):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        s[te] = m.decision_function(sc.transform(X[te]))
    return s


def valeur_ajoutee_sur_jetons(H: np.ndarray, prefixes: Sequence[Sequence[int]], y: np.ndarray,
                              groupes: Optional[np.ndarray] = None, plis: int = 5, C: float = 0.05, B: int = 1000,
                              seed: int = 0, marge: float = 0.05) -> Dict[str, Any]:
    """L'état caché à t prédit-il l'issue y mieux qu'un lecteur de jetons du même préfixe ? AUC hors échantillon des deux
    sondes logistiques (mêmes plis), écart et IC95 par bootstrap sur les groupes.

    VALEUR AJOUTÉE si la borne basse de l'écart > marge ; AUCUNE si l'IC95 ⊂ [−marge ; +marge] ; MOINS QUE LES JETONS si la
    borne haute < −marge ; sinon INDÉTERMINÉE. Le préfixe contient en principe toute l'information (l'état en est une
    fonction) : l'écart mesure ce que le calcul du modèle rend linéairement lisible au-delà d'un lecteur de jetons simple.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    groupes = np.arange(len(y)) if groupes is None else np.asarray(groupes)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < plis:
        return {"verdict": "NON TESTABLE (une seule classe ou trop peu d'exemples)"}
    sh = _oof_logistique(np.asarray(H, np.float32), y, groupes, plis, C)
    st = _oof_logistique(sac_de_jetons(prefixes), y, groupes, plis, C)
    ah, at = roc_auc_score(y, sh), roc_auc_score(y, st)
    rng = np.random.default_rng(seed)
    g_u = np.unique(groupes)
    par = {k: np.nonzero(groupes == k)[0] for k in g_u}
    bd, bh_, bt = [], [], []
    for _ in range(B):
        ix = np.concatenate([par[k] for k in rng.choice(g_u, len(g_u))])
        if len(np.unique(y[ix])) < 2:
            continue
        a1, a2 = roc_auc_score(y[ix], sh[ix]), roc_auc_score(y[ix], st[ix])
        bh_.append(a1)
        bt.append(a2)
        bd.append(a1 - a2)
    c = ci(bd)
    if c[0] > marge:
        v = "VALEUR AJOUTÉE"
    elif -marge <= c[0] and c[1] <= marge:
        v = "AUCUNE"
    elif c[1] < -marge:
        v = "MOINS QUE LES JETONS"
    else:
        v = "INDÉTERMINÉE"
    return {"auc_etats": float(ah), "auc_etats_ci": ci(bh_), "auc_jetons": float(at), "auc_jetons_ci": ci(bt),
            "ecart": float(ah - at), "ecart_ci": c, "verdict": v, "n": int(len(y))}

# --------------------------------------------------------------------------
# Moteur principal
# --------------------------------------------------------------------------

class TrajectoryBifurcationEngine:
    # Planchers absolus par mesure : sans eux, une ligne de base ~0 (premiers
    # tokens identiques, ex. "<think>" chez Qwen3) donne un seuil ~0 et une
    # bifurcation « détectée » dès t = min_steps. Valeurs heuristiques.
    DEFAULT_ABS_FLOOR = {"js": 0.05, "residual": 0.05, "residual_l2": 0.0}
    # Dénominateurs minimaux des recovery (nats) : en dessous, le rapport est instable
    MIN_JS_DEN = 0.05
    MIN_LL_DEN = 0.5

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        layers_to_track: Optional[List[int]] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.6,
        top_p: float = 1.0,
        topk_for_divergence: int = 256,
        seed: int = 0,
        trust_remote_code: bool = False,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        sampler: str = "gumbel",
        q_tsallis: float = 1.45,
        chat: Optional[bool] = None,
        forbid_special: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ):
        """sampler : "gumbel" (protocole SGZ, défaut) ou "multinomial" (v0.1.x). chat : None = gabarit de chat seulement si
        le tokenizer en a un et que le nom du modèle ne contient pas « Base ». forbid_special : None = vrai pour "gumbel"
        (longueur fixe, comme SGZ), faux pour "multinomial" (arrêt sur EOS, comme v0.1.x)."""
        if sampler not in ("gumbel", "multinomial"):
            raise ValueError("sampler : 'gumbel' ou 'multinomial'")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.topk_for_divergence = topk_for_divergence
        self.seed = seed
        self.sampler = sampler
        self.q = q_tsallis
        self.forbid_special = (sampler == "gumbel") if forbid_special is None else forbid_special

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if torch_dtype is None:
            # bf16 sur CPU est lent sans AVX512-BF16/AMX : float32 par défaut ;
            # sur GPU sans bf16 natif (T4, capacité < 8), fp16 comme SGZ
            if self.device.type == "cuda":
                major = torch.cuda.get_device_capability(self.device)[0]
                torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
            else:
                torch_dtype = torch.float32

        if model is None:
            print(f"[Core] Chargement de {model_name} sur {self.device}...")
            dtype_kw = (
                "dtype"
                if Version(transformers.__version__) >= Version("4.56")
                else "torch_dtype"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **{dtype_kw: torch_dtype},
            ).to(self.device)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.device = next(self.model.parameters()).device
        if chat is None:
            chat = bool(getattr(tokenizer, "chat_template", None)) and "base" not in model_name.lower()
        self.chat = chat

        self.num_layers = self.model.config.num_hidden_layers
        self.layers_to_track = layers_to_track or [
            0,
            self.num_layers // 2,
            self.num_layers - 1,
        ]

        # Qwen3 : <|im_end|> et <|endoftext|> terminent tous deux la génération
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        self.eos_token_ids = {
            int(e)
            for e in list(eos) + [self.tokenizer.eos_token_id]
            if e is not None
        }
        # Jetons interdits par l'échantillonneur SGZ : spéciaux, fin de texte
        forbid = set(getattr(tokenizer, "all_special_ids", None) or []) | set(self.eos_token_ids)
        forbid.discard(None)
        self._forbid = sorted(int(x) for x in forbid)
        self._n_tok = len(tokenizer)
        self._V = self.model.get_output_embeddings().weight.shape[0]
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
        npar = sum(p.numel() for p in self.model.parameters())
        self.batch_size = batch_size or (48 if npar < 3e9 else 32)

        print(f"[Core] Couches suivies : {self.layers_to_track}")
        print(f"[Core] Échantillonneur : {self.sampler}, T = {self.temperature}, top-p = {self.top_p} ; chat = {self.chat}")

    @classmethod
    def sgz(cls, model_name: str, **kw) -> "TrajectoryBifurcationEngine":
        """Moteur réglé comme les notebooks SGZ (A1c–B2) : Gumbel à nombres aléatoires communs, T = 0,6, sans top-p, prompt
        brut, jetons spéciaux interdits, Tsallis q = 1,45, lots de 48 (< 3 milliards de paramètres) ou 32."""
        kw.setdefault("sampler", "gumbel")
        kw.setdefault("temperature", 0.6)
        kw.setdefault("top_p", 1.0)
        kw.setdefault("chat", False)
        kw.setdefault("q_tsallis", 1.45)
        return cls(model_name=model_name, **kw)

    # ------------------------------------------------------------------
    # Hooks : capture et injection au même point (sortie brute de couche)
    # ------------------------------------------------------------------

    def _decoder_layers(self):
        # Accès aux couches (compatible Qwen / Llama-style, puis GPT-2-style)
        try:
            return self.model.model.layers
        except AttributeError:
            return self.model.transformer.h

    @staticmethod
    def _split_output(output):
        if isinstance(output, tuple):
            return output[0], output[1:]
        return output, None

    @contextlib.contextmanager
    def _capture_hooks(
        self, layers: List[int]
    ) -> Iterator[Dict[int, torch.Tensor]]:
        """Capture la sortie de chaque couche à la dernière position (ligne 0 du lot, comme v0.1.x).

        On n'utilise pas output_hidden_states : chez HF, hidden_states[-1]
        inclut la norme finale, ce qui ne correspond pas au point d'injection.
        Correspondance : couche l ici = hidden_states[l + 1] de HF pour l < L − 1.
        """
        captured: Dict[int, torch.Tensor] = {}

        def make_hook(l):
            def hook_fn(module, input, output):
                hidden, _ = self._split_output(output)
                captured[l] = hidden[0, -1, :].detach().float().cpu()

            return hook_fn

        layer_modules = self._decoder_layers()
        handles = [
            layer_modules[l].register_forward_hook(make_hook(l))
            for l in layers
        ]
        try:
            yield captured
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def _capture_hooks_batch(self, layers: Sequence[int], last: int = 1) -> Iterator[Dict[int, torch.Tensor]]:
        """Capture, pour tout le lot, la sortie de chaque couche aux `last` dernières positions (float32, sur l'appareil)."""
        captured: Dict[int, torch.Tensor] = {}

        def make_hook(l):
            def hook_fn(module, input, output):
                hidden, _ = self._split_output(output)
                captured[l] = hidden[:, -last:, :].detach().float()

            return hook_fn

        handles = [self._decoder_layers()[l].register_forward_hook(make_hook(l)) for l in layers]
        try:
            yield captured
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def _patch_hook(
        self, layer: int, position: int, vector: torch.Tensor
    ) -> Iterator[None]:
        """Remplace la sortie de `layer` à `position` lors de la 1re passe.

        Avec le KV-cache, la 1re passe (préremplissage) contient `position`,
        et les clés/valeurs des couches suivantes calculées à partir de l'état
        patché restent dans le cache : le patch agit sur toute la suite.
        """
        vector = vector.to(device=self.device, dtype=self.model.dtype)
        state = {"done": False}

        def hook_fn(module, input, output):
            if state["done"]:
                return output
            hidden, rest = self._split_output(output)
            if hidden.shape[1] <= position:
                raise RuntimeError(
                    f"Position {position} absente de la passe "
                    f"(longueur {hidden.shape[1]})"
                )
            new_hidden = hidden.clone()
            new_hidden[:, position, :] = vector
            state["done"] = True
            return new_hidden if rest is None else (new_hidden,) + rest

        handle = self._decoder_layers()[layer].register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    # ------------------------------------------------------------------
    # Échantillonnage
    # ------------------------------------------------------------------

    def _make_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return g

    def _mask(self, score: torch.Tensor) -> torch.Tensor:
        """Jetons interdits (spéciaux, fin de texte, hors vocabulaire du tokenizer), comme SGZ."""
        if self.forbid_special and self._forbid:
            score[:, self._forbid] = -float("inf")
        if self._V > self._n_tok:
            score[:, self._n_tok:] = -float("inf")
        return score

    def _nucleus(self, score: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Top-p appliqué au score de Gumbel (même noyau que la loi tempérée)."""
        if self.top_p >= 1.0:
            return score
        probs = F.softmax(logits / self.temperature, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        out = torch.cumsum(sp, dim=-1) > self.top_p
        out[..., 1:] = out[..., :-1].clone()
        out[..., 0] = False
        mask = torch.zeros_like(out).scatter(-1, si, out)
        return score.masked_fill(mask, -float("inf"))

    def _gumbel_next(self, logits: torch.Tensor, seeds: Sequence[int]) -> torch.Tensor:
        """argmax(z / T + Gumbel(graine)) sur les jetons permis (T ≤ 0 : argmax)."""
        if self.temperature <= 0:
            return torch.argmax(self._mask(logits.clone()), dim=-1)
        score = logits / self.temperature + gumbel_noise(self._V, seeds, logits.device)
        score = self._nucleus(score, logits)
        return torch.argmax(self._mask(score), dim=-1)

    def _sample(
        self, logits: torch.Tensor, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        """Échantillonneur v0.1.x (torch.multinomial)."""
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        probs_samp = F.softmax(logits / self.temperature, dim=-1)
        if self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs_samp, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > self.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / (
                sorted_probs.sum(dim=-1, keepdim=True) + 1e-12
            )
            choice = torch.multinomial(
                sorted_probs, num_samples=1, generator=generator
            )
            return sorted_idx.gather(-1, choice)
        return torch.multinomial(probs_samp, num_samples=1, generator=generator)

    def _topk(self, probs: torch.Tensor) -> TopK:
        k = min(self.topk_for_divergence, probs.shape[-1])
        top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
        top_probs_np = top_probs[0].cpu().numpy()
        top_indices_np = top_indices[0].cpu().numpy()
        remaining_mass = float(max(0.0, 1.0 - top_probs_np.sum()))
        return (top_indices_np, top_probs_np, remaining_mass)

    # ------------------------------------------------------------------
    # Génération par lots (protocole SGZ)
    # ------------------------------------------------------------------

    def _left_pad(self, seqs: Sequence[Sequence[int]]):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(list(s), dtype=torch.long)
            attn[j, L - len(s):] = 1
        return ids.to(self.device), attn.to(self.device), L

    @torch.no_grad()
    def generer_lot(
        self,
        contexts: Sequence[Sequence[int]],
        row_ids: Sequence[int],
        n_steps: int,
        seed_base: Optional[int] = None,
        step0: int = 0,
        layers: Sequence[int] = (),
        keep_steps: bool = False,
        mean_layers: Sequence[int] = (),
        force: Optional[Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        topk: int = 0,
        steps_to_keep: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Un lot de trajectoires (remplissage à gauche, KV-cache), échantillonneur Gumbel : graine du pas s de la ligne j =
        seed_base + 100003·row_ids[j] + step0 + s (identique à Gen.sample des notebooks SGZ).

        layers + keep_steps : états par pas (n, n_steps, d), ou seulement aux pas steps_to_keep (n, len(steps_to_keep), d) ;
        mean_layers : moyenne des états après lecture des jetons générés (une passe de plus pour lire le dernier).
        force(s, z, nxt) peut remplacer les jetons tirés (choc, préfixe forcé). lengths : jusqu'à la première fin de texte
        incluse (les jetons suivants sont générés mais n'ont pas de sens si les jetons spéciaux sont permis).
        """
        garder = None if steps_to_keep is None else set(int(x) for x in steps_to_keep)
        seed_base = self.seed if seed_base is None else seed_base
        n = len(contexts)
        rec = sorted(set(layers if keep_steps else ()) | set(mean_layers))
        toks = np.zeros((n, n_steps), np.int64)
        Hs, Ht = np.zeros((n, n_steps), np.float32), np.zeros((n, n_steps), np.float32)
        steps = {l: [] for l in (layers if keep_steps else ())}
        acc = {l: None for l in mean_layers}
        tk: List[List[TopK]] = [[] for _ in range(n)] if topk else None
        ids, attn, _ = self._left_pad(contexts)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        past = None
        with self._capture_hooks_batch(rec) as cap:
            for s in range(n_steps + (1 if mean_layers else 0)):
                out = self.model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
                if s >= 1:
                    for l in mean_layers:
                        h = cap[l][:, -1, :]
                        acc[l] = h.clone() if acc[l] is None else acc[l] + h
                if s == n_steps:
                    break
                if garder is None or s in garder:
                    for l in steps:
                        steps[l].append(cap[l][:, -1, :].cpu().numpy().astype(np.float16))
                past = out.past_key_values
                z = out.logits[:, -1, :].float()
                if not torch.isfinite(z).all():
                    raise FloatingPointError(f"logits non finis au pas {s}")
                e = entropies_from_logits(z, [1.0, self.q])
                Hs[:, s], Ht[:, s] = e[1.0].cpu().numpy(), e[self.q].cpu().numpy()
                if topk:
                    p = F.softmax(z, dim=-1)
                    tp, ti = torch.topk(p, k=min(topk, p.shape[-1]), dim=-1)
                    tp, ti = tp.cpu().numpy(), ti.cpu().numpy()
                    for j in range(n):
                        tk[j].append((ti[j], tp[j], float(max(0.0, 1.0 - tp[j].sum()))))
                seeds = [seed_base + PAS_GRAINE * int(row_ids[j]) + step0 + s for j in range(n)]
                nxt = self._gumbel_next(z, seeds)
                if force is not None:
                    nxt = force(s, z, nxt)
                toks[:, s] = nxt.cpu().numpy()
                ids = nxt[:, None].to(self.device)
                attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=self.device)], dim=1)
                pos = pos[:, -1:] + 1
        fin = np.isin(toks, list(self.eos_token_ids))
        lengths = np.where(fin.any(1), fin.argmax(1) + 1, n_steps)
        return {"tokens": toks, "entropy_shannon": Hs, "entropy_tsallis": Ht, "lengths": lengths,
                "hidden_steps": {l: np.stack(v, 1) for l, v in steps.items()},
                "hidden_mean": {l: (acc[l] / n_steps).cpu().numpy() for l in mean_layers}, "topk": tk}

    def encoder(self, prompt: str, system_prompt: Optional[str] = None,
                chat_template_kwargs: Optional[Dict[str, Any]] = None) -> List[int]:
        """Jetons du prompt : gabarit de chat si self.chat, sinon tokenizer.encode(prompt) (comme SGZ)."""
        if self.chat:
            return self.build_prompt_ids(prompt, system_prompt, chat_template_kwargs)[0].cpu().tolist()
        return list(self.tokenizer.encode(prompt))

    def generer_ensemble(
        self,
        prompts: Sequence[str],
        n: int,
        n_steps: Optional[int] = None,
        seed_base: Optional[int] = None,
        row_stride: int = 256,
        layers: Sequence[int] = (),
        keep_steps: bool = False,
        mean_layers: Sequence[int] = (),
        topk: int = 0,
        batch_size: Optional[int] = None,
        prompt_ids: Optional[Sequence[Sequence[int]]] = None,
        progress: bool = True,
        steps_to_keep: Optional[Sequence[int]] = None,
    ) -> Ensemble:
        """n trajectoires par prompt, par lots. Ligne (p, j) : identifiant de bruit p·row_stride + j (B2 : 256), lots formés
        dans l'ordre (p, j) comme B2 ; même seed_base, même modèle et même taille de lot → mêmes jetons que B2."""
        if self.sampler != "gumbel":
            raise ValueError("generer_ensemble utilise l'échantillonneur 'gumbel'")
        if n > row_stride:
            raise ValueError("n doit être ≤ row_stride (identifiants de bruit distincts)")
        n_steps = self.max_new_tokens if n_steps is None else n_steps
        enc = [list(x) for x in prompt_ids] if prompt_ids is not None else [self.encoder(p) for p in prompts]
        bs = batch_size or self.batch_size
        jobs = [(p, j) for p in range(len(enc)) for j in range(n)]
        parts = []
        it = range(0, len(jobs), bs)
        for b0 in (tqdm(it, desc=f"Ensemble {len(enc)}×{n}") if progress else it):
            lot = jobs[b0:b0 + bs]
            parts.append(self.generer_lot([enc[p] for p, _ in lot], [p * row_stride + j for p, j in lot], n_steps,
                                          seed_base=seed_base, layers=layers, keep_steps=keep_steps,
                                          mean_layers=mean_layers, topk=topk, steps_to_keep=steps_to_keep))
        cat = lambda k: np.concatenate([x[k] for x in parts])
        return Ensemble(
            prompt_ids=enc, prompt_of=np.array([p for p, _ in jobs]), row_ids=np.array([p * row_stride + j for p, j in jobs]),
            tokens=cat("tokens"), entropy_shannon=cat("entropy_shannon"), entropy_tsallis=cat("entropy_tsallis"),
            hidden_steps={l: np.concatenate([x["hidden_steps"][l] for x in parts]) for l in (layers if keep_steps else ())},
            hidden_mean={l: np.concatenate([x["hidden_mean"][l] for x in parts]) for l in mean_layers},
            topk=sum((x["topk"] for x in parts), []) if topk else None,
            kept_steps=None if steps_to_keep is None else [int(x) for x in steps_to_keep], lengths=cat("lengths"),
            seed_base=self.seed if seed_base is None else seed_base, temperature=self.temperature, q=self.q)

    @torch.no_grad()
    def rejouer_moyennes(self, contexts: Sequence[Sequence[int]], suites: np.ndarray, layers: Sequence[int],
                         batch_size: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Passe complète (sans cache) sur contexte + suite : moyenne, sur les jetons de la suite, de la sortie de chaque
        couche. Identique à la représentation R_h de B2 (hidden_states[c] de HF = couche c − 1 ici, pour c < L)."""
        bs = batch_size or self.batch_size
        T = suites.shape[1]
        out = {l: [] for l in layers}
        for b0 in range(0, len(contexts), bs):
            loc = range(b0, min(len(contexts), b0 + bs))
            ids, attn, _ = self._left_pad([list(contexts[i]) + suites[i].tolist() for i in loc])
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            with self._capture_hooks_batch(layers, last=T) as cap:
                self.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
            for l in layers:
                out[l].append(cap[l].mean(1).cpu().numpy())
        return {l: np.concatenate(v) for l, v in out.items()}

    @torch.no_grad()
    def ensemble_choc(self, prompts: Sequence[str], t0: int, n_steps: int, rang_lo: int = 1000, rang_hi: int = 5000,
                      seed_base: Optional[int] = None, seed_rangs: Optional[int] = None,
                      batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Protocole A1 : pour chaque prompt, un témoin et un choc générés ensemble avec le même bruit ; au pas t0, le choc
        reçoit le jeton de rang r (tiré dans [rang_lo, rang_hi], jamais celui du témoin) dans la loi brute du modèle."""
        seed_base = self.seed if seed_base is None else seed_base
        rng = np.random.default_rng((seed_base + 5) if seed_rangs is None else seed_rangs)
        enc = [self.encoder(p) for p in prompts]
        P = len(enc)
        rangs = rng.integers(rang_lo, rang_hi + 1, P)
        res = {"temoin": np.zeros((P, n_steps), np.int64), "choc": np.zeros((P, n_steps), np.int64),
               "S_temoin": np.zeros((P, n_steps), np.float32), "S_choc": np.zeros((P, n_steps), np.float32),
               "X": np.zeros(P, np.int64), "rang": np.zeros(P, np.int64), "surprise": np.zeros(P)}
        bs = max(1, (batch_size or self.batch_size) // 2)
        for b0 in range(0, P, bs):
            loc = list(range(b0, min(P, b0 + bs)))
            m = len(loc)

            def force(s, z, nxt, loc=loc, m=m):
                if s < t0:
                    nxt[m:] = nxt[:m]
                elif s == t0:
                    zz = self._mask(z[m:].clone())
                    lp = torch.log_softmax(z[m:].double(), dim=-1)
                    order = torch.argsort(zz, dim=-1, descending=True)
                    for j, i in enumerate(loc):
                        rk = int(min(rangs[i], order.shape[1] - 2))
                        if int(order[j, rk]) == int(nxt[j]):
                            rk += 1
                        x = int(order[j, rk])
                        res["X"][i], res["rang"][i], res["surprise"][i] = x, rk, -float(lp[j, x])
                        nxt[m + j] = x
                return nxt

            o = self.generer_lot([enc[i] for i in loc] * 2, loc * 2, n_steps, seed_base=seed_base, force=force)
            res["temoin"][loc], res["choc"][loc] = o["tokens"][:m], o["tokens"][m:]
            res["S_temoin"][loc], res["S_choc"][loc] = o["entropy_tsallis"][:m], o["entropy_tsallis"][m:]
        return res

    # ------------------------------------------------------------------
    # Génération une trajectoire à la fois (API v0.1.x)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        seed: Optional[int],
        record_layers: Optional[List[int]] = None,
        patch: Optional[Tuple[int, int, torch.Tensor]] = None,
        step0: int = 0,
    ) -> Dict[str, Any]:
        """Boucle d'échantillonnage commune (ensemble et continuations).

        Entropie et top-k sont mesurés sur la distribution du modèle (T = 1) ;
        l'échantillonnage utilise temperature / top_p. Avec "gumbel", la graine du
        pas s est seed_base + 100003·seed + step0 + s (seed = identifiant de ligne).
        """
        record_layers = record_layers or []
        generator = self._make_generator(seed) if self.sampler == "multinomial" else None
        new_tokens: List[int] = []
        entropies: List[float] = []
        tsallis: List[float] = []
        topk_dists: List[TopK] = []
        hiddens: Dict[int, List[np.ndarray]] = {l: [] for l in record_layers}
        row = 0 if seed is None else int(seed)

        with contextlib.ExitStack() as stack:
            captured = stack.enter_context(self._capture_hooks(record_layers))
            if patch is not None:
                stack.enter_context(self._patch_hook(*patch))

            past = None
            cur = input_ids
            for s in range(max_new_tokens):
                outputs = self.model(
                    input_ids=cur,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = outputs.past_key_values
                logits = outputs.logits[:, -1, :].float()
                probs = F.softmax(logits, dim=-1)

                entropies.append(
                    float(-torch.sum(probs * torch.log(probs + 1e-12)).item())
                )
                tsallis.append(float(entropies_from_logits(logits, [self.q])[self.q][0]))
                topk_dists.append(self._topk(probs))
                for l in record_layers:
                    hiddens[l].append(captured[l].numpy())

                if self.sampler == "gumbel":
                    next_token = self._gumbel_next(logits, [self.seed + PAS_GRAINE * row + step0 + s]).view(1, 1)
                else:
                    next_token = self._sample(logits, generator)
                token = int(next_token.item())
                new_tokens.append(token)
                if token in self.eos_token_ids:
                    break
                cur = next_token.view(1, 1)

        return {
            "new_tokens": new_tokens,
            "entropies": entropies,
            "tsallis": tsallis,
            "topk": topk_dists,
            "hidden_states": hiddens,
        }

    def build_prompt_ids(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Jetons du prompt (1, n) : gabarit de chat si self.chat, sinon prompt brut."""
        if not self.chat:
            return torch.tensor([list(self.tokenizer.encode(prompt))], device=self.device)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs["input_ids"].to(self.device)

    def _generate_one_trajectory(
        self, input_ids: torch.Tensor, seed: Optional[int] = None
    ) -> Trajectory:
        out = self._decode_loop(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            seed=seed,
            record_layers=self.layers_to_track,
        )
        prompt_ids = input_ids[0].cpu().tolist()
        return Trajectory(
            token_ids=prompt_ids + out["new_tokens"],
            text=self.tokenizer.decode(
                out["new_tokens"], skip_special_tokens=True
            ),
            prompt_len=len(prompt_ids),
            seed=seed,
            entropy_shannon=out["entropies"],
            hidden_states=out["hidden_states"],
            topk_distributions=out["topk"],
            entropy_tsallis=out["tsallis"],
        )

    def generate_ensemble(
        self,
        prompt: str,
        n_trajectories: int = 16,
        system_prompt: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> List[Trajectory]:
        """Liste de Trajectory (API v0.1.x). Avec "gumbel", la trajectoire i a l'identifiant de bruit base + i."""
        input_ids = self.build_prompt_ids(
            prompt, system_prompt, chat_template_kwargs
        )
        base_seed = self.seed if seed is None else seed

        trajectories = []
        for i in tqdm(
            range(n_trajectories), desc=f"Ensemble n={n_trajectories}"
        ):
            trajectories.append(
                self._generate_one_trajectory(input_ids, seed=base_seed + i)
            )
        return trajectories

    # ------------------------------------------------------------------
    # Divergences
    # ------------------------------------------------------------------

    @staticmethod
    def js_divergence_topk(
        dist_a: TopK,
        dist_b: TopK,
        eps: float = 1e-12,
    ) -> float:
        """JS (nats, borne ln 2) entre deux distributions tronquées au top-k.

        La masse hors top-k de chaque distribution est placée dans une case
        « reste » commune, au lieu de renormaliser le top-k (ce qui gonflait
        la divergence). Approximation : l'erreur est bornée par les masses
        résiduelles. Un token présent dans un seul top-k reçoit 0 de l'autre
        côté, alors que sa vraie probabilité est dans le reste.
        """
        idx_a, p_a, rem_a = dist_a
        idx_b, p_b, rem_b = dist_b

        all_idx = np.union1d(idx_a, idx_b)
        pa = np.zeros(len(all_idx) + 1, dtype=np.float64)
        pb = np.zeros(len(all_idx) + 1, dtype=np.float64)
        pa[np.searchsorted(all_idx, idx_a)] = p_a
        pb[np.searchsorted(all_idx, idx_b)] = p_b
        pa[-1] = max(rem_a, 0.0)
        pb[-1] = max(rem_b, 0.0)

        pa = pa / (pa.sum() + eps)
        pb = pb / (pb.sum() + eps)
        m = 0.5 * (pa + pb)

        def kl(p, q):
            nz = p > 0
            return float(np.sum(p[nz] * np.log(p[nz] / (q[nz] + eps))))

        return max(0.0, 0.5 * (kl(pa, m) + kl(pb, m)))

    @staticmethod
    def first_diff_step(
        src: Trajectory, tgt: Trajectory, t_patch: int
    ) -> Optional[int]:
        a, b = src.generated_ids[:t_patch], tgt.generated_ids[:t_patch]
        return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)

    @classmethod
    def classify_pair(cls, src: Trajectory, tgt: Trajectory, t_patch: int) -> str:
        """Classement v0.1.x : negative_control, positive_control ou informative."""
        first_diff = cls.first_diff_step(src, tgt, t_patch)
        if first_diff is None:
            return "negative_control"
        if first_diff == t_patch - 1:
            return "positive_control"
        return "informative"

    @classmethod
    def classify_pair_detail(cls, src: Trajectory, tgt: Trajectory, t_patch: int) -> str:
        """Comme classify_pair, « informative » scindé : informative_context (dernier jeton du préfixe identique, le patch
        ne transmet que le contexte) ou informative_token (dernier jeton différent : le patch transmet aussi son identité)."""
        kind = cls.classify_pair(src, tgt, t_patch)
        if kind != "informative":
            return kind
        same = src.generated_ids[t_patch - 1] == tgt.generated_ids[t_patch - 1]
        return "informative_context" if same else "informative_token"

    @staticmethod
    def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        return float(1.0 - np.dot(u, v) / (nu * nv))

    def compute_divergences(
        self, trajectories: List[Trajectory]
    ) -> Dict[str, Any]:
        n = len(trajectories)
        if n < 2:
            raise ValueError("Au moins 2 trajectoires requises")

        min_len = min(len(t.entropy_shannon) for t in trajectories)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        residual_dists = {l: [] for l in self.layers_to_track}
        residual_cos = {l: [] for l in self.layers_to_track}
        js_over_time = []
        js_diff_prefix = []
        frac_same_prefix = []

        for t in range(min_len):
            # À l'étape t, l'état dépend des tokens générés 0..t-1
            same = [
                trajectories[i].generated_ids[:t]
                == trajectories[j].generated_ids[:t]
                for i, j in pairs
            ]
            frac_same_prefix.append(float(np.mean(same)))

            for l in self.layers_to_track:
                vecs = [traj.hidden_states[l][t] for traj in trajectories]
                residual_dists[l].append(
                    float(
                        np.mean(
                            [np.linalg.norm(vecs[i] - vecs[j]) for i, j in pairs]
                        )
                    )
                )
                residual_cos[l].append(
                    float(
                        np.mean(
                            [
                                self._cosine_distance(vecs[i], vecs[j])
                                for i, j in pairs
                            ]
                        )
                    )
                )

            js_t = [
                self.js_divergence_topk(
                    trajectories[i].topk_distributions[t],
                    trajectories[j].topk_distributions[t],
                )
                for i, j in pairs
            ]
            js_over_time.append(float(np.mean(js_t)))
            dj = [x for x, s in zip(js_t, same) if not s]
            js_diff_prefix.append(float(np.mean(dj)) if dj else 0.0)

        ent_matrix = np.array(
            [t.entropy_shannon[:min_len] for t in trajectories]
        )
        js_arr, frac = np.array(js_over_time), np.array(frac_same_prefix)
        corr = (float(np.corrcoef(js_arr, 1 - frac)[0, 1])
                if min_len > 2 and np.std(js_arr) > 0 and np.std(frac) > 0 else None)

        return {
            "min_len": min_len,
            "residual_dist_over_time": {
                l: np.array(v) for l, v in residual_dists.items()
            },
            "residual_cos_over_time": {
                l: np.array(v) for l, v in residual_cos.items()
            },
            "js_over_time": js_arr,
            "frac_pairs_same_prefix": frac,
            # JS moyenne sur les seules paires dont les préfixes diffèrent : une JS
            # moyenne qui monte peut ne refléter que la part de paires divergentes
            "js_paires_divergentes": np.array(js_diff_prefix),
            # Critère C1 de v0.1.5 : ≥ 0,9 → la « bifurcation » JS suit les jetons
            "corr_js_divergence_jetons": corr,
            "entropy_mean": ent_matrix.mean(axis=0),
            "entropy_std": ent_matrix.std(axis=0),
            "n_trajectories": n,
        }

    def detect_bifurcation(
        self,
        divergence_data: Dict[str, Any],
        measure: str = "js",
        layer: Optional[int] = None,
        threshold_ratio: float = 2.5,
        min_steps: int = 4,
        abs_floor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """measure : "js", "residual" (cosinus) ou "residual_l2".

        Attention : une bifurcation JS coïncide le plus souvent avec la divergence
        des jetons (corr_js_divergence_jetons) ; pour une structure interne, voir
        analyser_regimes et valeur_ajoutee_sur_jetons.
        """
        if measure == "js":
            series = divergence_data["js_over_time"]
            used = "js"
        else:
            if layer is None:
                layer = self.layers_to_track[len(self.layers_to_track) // 2]
            if measure == "residual_l2":
                series = divergence_data["residual_dist_over_time"][layer]
                used = f"residual_l2_layer_{layer}"
            else:
                series = divergence_data["residual_cos_over_time"][layer]
                used = f"residual_cos_layer_{layer}"
        if abs_floor is None:
            abs_floor = self.DEFAULT_ABS_FLOOR.get(measure, 0.0)

        if len(series) < min_steps + 1:
            return {
                "bifurcation_detected": False,
                "t_star": None,
                "measure": used,
            }

        baseline = float(np.mean(series[:min_steps]))
        threshold = max(baseline * threshold_ratio, abs_floor)

        t_star = None
        for t in range(min_steps, len(series)):
            if series[t] > threshold:
                t_star = t
                break

        return {
            "bifurcation_detected": t_star is not None,
            "t_star": t_star,
            "measure": used,
            "baseline": baseline,
            "threshold": threshold,
            # Le plancher l'emporte : le seuil relatif seul serait trivial
            "baseline_degenerate": baseline * threshold_ratio < abs_floor,
            "final_value": float(series[-1]),
        }

    # ------------------------------------------------------------------
    # Activation Patching (hooks)
    # ------------------------------------------------------------------

    def continue_from_prefix(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int = 32,
        seed: Optional[int] = None,
        patch: Optional[Tuple[int, int, torch.Tensor]] = None,
        step0: int = 0,
    ) -> Continuation:
        """Continue `prefix_ids`, éventuellement avec patch (couche, pos, vecteur).

        Avec "gumbel", seed = identifiant de ligne et step0 = indice du premier jeton généré : même (seed, step0) → même
        bruit à chaque position, pour la version propre comme pour la version patchée."""
        out = self._decode_loop(
            prefix_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            seed=seed,
            patch=patch,
            step0=step0,
        )
        return Continuation(
            token_ids=out["new_tokens"],
            text=self.tokenizer.decode(
                out["new_tokens"], skip_special_tokens=True
            ),
            first_distribution=out["topk"][0],
        )

    def activation_patch_with_hook(
        self,
        prefix_ids: torch.Tensor,
        source_hidden: torch.Tensor,
        layer: int,
        max_new_tokens: int = 32,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], str]:
        """Compatibilité v0.1.3 : patch à la dernière position du préfixe.

        Renvoie (tokens générés, texte généré), sans le préfixe.
        """
        cont = self.continue_from_prefix(
            prefix_ids,
            max_new_tokens=max_new_tokens,
            seed=seed,
            patch=(layer, prefix_ids.shape[1] - 1, source_hidden),
        )
        return cont.token_ids, cont.text

    @staticmethod
    def _js_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """JS exacte (nats) ligne par ligne entre distributions complètes."""
        m = 0.5 * (p + q)
        xlogy = torch.special.xlogy
        kl_pm = (xlogy(p, p) - xlogy(p, m)).sum(-1)
        kl_qm = (xlogy(q, q) - xlogy(q, m)).sum(-1)
        return (0.5 * (kl_pm + kl_qm)).clamp_min(0.0)

    @torch.no_grad()
    def forced_patch_effect(
        self,
        src: Trajectory,
        tgt: Trajectory,
        t_patch: int,
        layer: int,
        vector: Optional[np.ndarray] = None,
        horizon: int = 16,
    ) -> Dict[str, Any]:
        """Effet du patch le long de la continuation de la source (forçage).

        Trois passes sur préfixe + src.generated_ids[t_patch:t_patch+K] :
        cible propre, cible patchée (état de la source par défaut, ou `vector`
        pour un témoin) et source. Distributions exactes sur tout le
        vocabulaire. À la position patchée, les clés/valeurs des couches
        <= layer restent celles de la cible : pour k >= 1 l'état est hybride.
        Rapports à None si le dénominateur est trop petit (MIN_JS_DEN, MIN_LL_DEN).
        """
        if src.prompt_len != tgt.prompt_len:
            raise ValueError("Source et cible n'ont pas le même prompt")
        if vector is None:
            vector = src.hidden_states[layer][t_patch]
        prefix_end = tgt.prompt_len + t_patch
        position = prefix_end - 1
        forced = src.generated_ids[t_patch : t_patch + horizon]
        k_steps = len(forced)
        if k_steps == 0:
            raise ValueError("Aucun token de la source à forcer")

        def log_probs(prefix, patch=None):
            ids = torch.tensor([prefix + forced], device=self.device)
            with contextlib.ExitStack() as stack:
                if patch is not None:
                    stack.enter_context(self._patch_hook(*patch))
                logits = self.model(input_ids=ids, use_cache=False).logits
            return F.log_softmax(
                logits[0, position : position + k_steps].double(), dim=-1
            )

        lp_clean = log_probs(tgt.token_ids[:prefix_end])
        lp_patch = log_probs(
            tgt.token_ids[:prefix_end],
            (layer, position, torch.from_numpy(np.asarray(vector, dtype=np.float32))),
        )
        lp_src = log_probs(src.token_ids[:prefix_end])

        js_cs = self._js_rows(lp_clean.exp(), lp_src.exp())
        js_ps = self._js_rows(lp_patch.exp(), lp_src.exp())
        per_step = [
            float(1.0 - b / a) if a > 1e-6 else None
            for a, b in zip(js_cs.tolist(), js_ps.tolist())
        ]

        tok = torch.tensor(forced, device=lp_src.device).unsqueeze(1)
        ll_clean = float(lp_clean.gather(1, tok).sum())
        ll_patch = float(lp_patch.gather(1, tok).sum())
        ll_src = float(lp_src.gather(1, tok).sum())

        den_js = float(js_cs.sum())
        den_ll = ll_src - ll_clean
        return {
            "horizon": k_steps,
            "recovery_per_step": per_step,
            "recovery_multistep": (
                1.0 - float(js_ps.sum()) / den_js if den_js > self.MIN_JS_DEN else None
            ),
            "recovery_loglik": (
                (ll_patch - ll_clean) / den_ll if abs(den_ll) > self.MIN_LL_DEN else None
            ),
            "js_clean_source": js_cs.tolist(),
            "js_patched_source": js_ps.tolist(),
            "ll": {"clean": ll_clean, "patched": ll_patch, "source": ll_src},
        }

    def choose_patch_time(
        self,
        trajectories: List[Trajectory],
        t_min: int,
        n_pairs: int,
        kind: str = "informative",
    ) -> Optional[int]:
        """Premier t >= t_min offrant au moins n_pairs paires du type `kind`
        ("informative", "informative_context" ou "informative_token").

        À t* même, les paires ne diffèrent souvent qu'au dernier token
        (témoins positifs) : patcher là ne teste rien.
        """
        n = len(trajectories)
        lengths = [len(t.entropy_shannon) for t in trajectories]
        cls = self.classify_pair if kind == "informative" else self.classify_pair_detail
        for t in range(max(1, t_min), max(lengths)):
            count = sum(
                1
                for i in range(n)
                for j in range(i + 1, n)
                if min(lengths[i], lengths[j]) > t
                and cls(trajectories[i], trajectories[j], t) == kind
            )
            if count >= n_pairs:
                return t
        return None

    def activation_patch_test(
        self,
        trajectories: List[Trajectory],
        source_idx: int,
        target_idx: int,
        t_patch: int,
        layer: int,
        max_continuation_tokens: int = 32,
        seed: Optional[int] = None,
        overlap_threshold: float = 0.55,
        forced_horizon: int = 16,
    ) -> PatchResult:
        src = trajectories[source_idx]
        tgt = trajectories[target_idx]

        if (
            t_patch >= len(src.hidden_states[layer])
            or t_patch >= len(tgt.hidden_states[layer])
        ):
            raise ValueError(
                f"t_patch={t_patch} dépasse la longueur des trajectoires"
            )
        if src.prompt_len != tgt.prompt_len:
            raise ValueError("Source et cible n'ont pas le même prompt")

        # L'état à l'étape t est celui de la dernière position de
        # prompt + tokens générés[:t]
        prefix_end = tgt.prompt_len + t_patch
        position = prefix_end - 1
        prefix_ids = torch.tensor(
            [tgt.token_ids[:prefix_end]], device=self.device
        )
        source_hidden = torch.from_numpy(np.asarray(src.hidden_states[layer][t_patch], dtype=np.float32))

        if seed is None:
            seed = self.seed + 7919 * (source_idx + 1) + target_idx
        # Même graine (et, avec "gumbel", même bruit à chaque position) : toute
        # différence entre les deux continuations vient du patch
        clean = self.continue_from_prefix(
            prefix_ids, max_continuation_tokens, seed=seed, step0=t_patch
        )
        patched = self.continue_from_prefix(
            prefix_ids,
            max_continuation_tokens,
            seed=seed,
            patch=(layer, position, source_hidden),
            step0=t_patch,
        )

        orig_set = set(clean.text.split())
        patch_set = set(patched.text.split())
        union = orig_set | patch_set
        # Deux continuations vides (EOS immédiat) sont identiques
        overlap = len(orig_set & patch_set) / len(union) if union else 1.0

        first_div = next(
            (
                k
                for k, (a, b) in enumerate(
                    zip(clean.token_ids, patched.token_ids)
                )
                if a != b
            ),
            None,
        )
        if first_div is None and len(clean.token_ids) != len(
            patched.token_ids
        ):
            first_div = min(len(clean.token_ids), len(patched.token_ids))

        p_src = src.topk_distributions[t_patch]
        js_ts = self.js_divergence_topk(clean.first_distribution, p_src)
        js_ps = self.js_divergence_topk(patched.first_distribution, p_src)
        js_pt = self.js_divergence_topk(
            patched.first_distribution, clean.first_distribution
        )
        recovery = 1.0 - js_ps / js_ts if js_ts > 1e-6 else None
        pair_kind = self.classify_pair(src, tgt, t_patch)
        forced = (
            self.forced_patch_effect(src, tgt, t_patch, layer, horizon=forced_horizon)
            if forced_horizon > 0
            else {}
        )

        return PatchResult(
            source_traj_idx=source_idx,
            target_traj_idx=target_idx,
            layer=layer,
            t_patch=t_patch,
            original_continuation=clean.text,
            patched_continuation=patched.text,
            final_token_overlap=float(overlap),
            success=overlap < overlap_threshold,
            seed=seed,
            pair_kind=pair_kind,
            prefixes_identical=pair_kind == "negative_control",
            first_divergence_step=first_div,
            js_target_source=js_ts,
            js_patched_source=js_ps,
            js_patched_target=js_pt,
            recovery=recovery,
            js_clean_vs_recorded=self.js_divergence_topk(
                clean.first_distribution, tgt.topk_distributions[t_patch]
            ),
            first_diff_step=self.first_diff_step(src, tgt, t_patch),
            last_token_same=(
                src.generated_ids[t_patch - 1] == tgt.generated_ids[t_patch - 1]
                if t_patch >= 1
                else None
            ),
            recovery_multistep=forced.get("recovery_multistep"),
            recovery_loglik=forced.get("recovery_loglik"),
            recovery_per_step=forced.get("recovery_per_step"),
            forced_horizon=forced.get("horizon", 0),
            pair_detail=self.classify_pair_detail(src, tgt, t_patch),
        )

    def reversibility_analysis(
        self,
        trajectories: List[Trajectory],
        t_patch: int,
        layer: Optional[int] = None,
        n_pairs: int = 5,
        n_controls: int = 1,
        max_continuation_tokens: int = 32,
        seed: Optional[int] = None,
        forced_horizon: int = 16,
        detail: bool = False,
    ) -> List[PatchResult]:
        """Patch source → cible sur des paires tirées au hasard.

        Paires informatives : préfixes différents avant la position patchée.
        Témoins négatifs : préfixes identiques (patch nul ; continuations
        identiques attendues). Témoins positifs : seul le dernier token du
        préfixe diffère (recovery = 1 attendu). Jusqu'à n_controls de chaque.
        detail=True : n_pairs paires « contexte » et n_pairs paires « jeton ».
        """
        if layer is None:
            layer = self.layers_to_track[len(self.layers_to_track) // 2]
        if layer == self.num_layers - 1:
            # Seuls la norme finale et lm_head suivent : recovery = 1 trivial
            print("[Patch] Attention : dernière couche → recovery = 1 par construction")

        rng = np.random.default_rng(self.seed if seed is None else seed)
        n = len(trajectories)
        ordered = [(i, j) for i in range(n) for j in range(n) if i != j]
        rng.shuffle(ordered)

        if detail:
            quota = {"informative_context": n_pairs, "informative_token": n_pairs,
                     "positive_control": n_controls, "negative_control": n_controls}
            cls = self.classify_pair_detail
        else:
            quota = {"informative": n_pairs, "positive_control": n_controls, "negative_control": n_controls}
            cls = self.classify_pair
        selected: Dict[str, List[Tuple[int, int]]] = {k: [] for k in quota}
        seen = set()
        for i, j in ordered:
            if frozenset((i, j)) in seen:
                continue
            seen.add(frozenset((i, j)))
            if min(len(trajectories[i].hidden_states[layer]),
                   len(trajectories[j].hidden_states[layer])) <= t_patch:
                continue
            kind = cls(trajectories[i], trajectories[j], t_patch)
            if len(selected[kind]) < quota[kind]:
                selected[kind].append((i, j))

        results = []
        for i, j in sum(selected.values(), []):
            try:
                results.append(
                    self.activation_patch_test(
                        trajectories=trajectories,
                        source_idx=i,
                        target_idx=j,
                        t_patch=t_patch,
                        layer=layer,
                        max_continuation_tokens=max_continuation_tokens,
                        forced_horizon=forced_horizon,
                    )
                )
            except (ValueError, RuntimeError) as e:
                print(f"[Patch] Échec paire ({i},{j}) : {e}")
        return results

    # ------------------------------------------------------------------
    # Pipeline complet
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_patches(patch_results: List[PatchResult]) -> Dict[str, Any]:
        def mean(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None

        def median(values):
            values = [v for v in values if v is not None]
            return float(np.median(values)) if values else None

        info = [r for r in patch_results if r.pair_kind == "informative"]
        pos = [r for r in patch_results if r.pair_kind == "positive_control"]
        neg = [r for r in patch_results if r.pair_kind == "negative_control"]
        out = {
            "n_patch_pairs": len(info),
            "n_positive_controls": len(pos),
            "n_negative_controls": len(neg),
            "patch_success_rate": mean([r.success for r in info]),
            "patch_mean_overlap": mean([r.final_token_overlap for r in info]),
            "patch_mean_recovery": mean([r.recovery for r in info]),
            "patch_mean_recovery_multistep": mean(
                [r.recovery_multistep for r in info]
            ),
            "patch_mean_recovery_loglik": mean([r.recovery_loglik for r in info]),
            # Attendu ~1 (contexte identique sauf la position patchée)
            "positive_control_mean_recovery": mean([r.recovery for r in pos]),
            # Attendu = 1 (patch nul, même graine → continuations identiques)
            "negative_control_mean_overlap": mean(
                [r.final_token_overlap for r in neg]
            ),
            "max_js_clean_vs_recorded": (
                float(max(r.js_clean_vs_recorded for r in patch_results))
                if patch_results
                else None
            ),
        }
        # v0.2 : médianes séparées « contexte » / « jeton » (le recovery des paires
        # « jeton » mesure surtout l'identité du dernier jeton)
        for d in ("informative_context", "informative_token"):
            sub = [r for r in info if r.pair_detail == d]
            out[f"n_{d}"] = len(sub)
            out[f"median_recovery_multistep_{d}"] = median([r.recovery_multistep for r in sub])
            out[f"median_recovery_loglik_{d}"] = median([r.recovery_loglik for r in sub])
        return out

    def analyze_prompt(
        self,
        prompt: str,
        n_trajectories: int = 12,
        system_prompt: Optional[str] = None,
        run_patching: bool = True,
        n_patch_pairs: int = 4,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        forced_horizon: int = 16,
        require_bifurcation: bool = True,
    ) -> BifurcationResult:
        print(f"\n[Core] Analyse : {prompt[:90]}...")

        trajectories = self.generate_ensemble(
            prompt,
            n_trajectories=n_trajectories,
            system_prompt=system_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        div = self.compute_divergences(trajectories)

        bif_js = self.detect_bifurcation(div, measure="js")
        bif_res = self.detect_bifurcation(div, measure="residual")

        patch_results = []
        t_star = bif_js.get("t_star")
        if t_star is None:
            t_star = bif_res.get("t_star")

        t_patch = None
        if run_patching and (t_star is not None or not require_bifurcation):
            t_patch = self.choose_patch_time(
                trajectories,
                t_min=t_star if t_star is not None else 1,
                n_pairs=n_patch_pairs,
            )
            if t_patch is None:
                print("[Core] Aucun t avec assez de paires informatives")
            else:
                print(
                    f"[Core] t*={t_star} → test de réversibilité à t={t_patch}"
                )
                patch_results = self.reversibility_analysis(
                    trajectories=trajectories,
                    t_patch=t_patch,
                    n_pairs=n_patch_pairs,
                    forced_horizon=forced_horizon,
                )

        summary = {
            "min_len": div["min_len"],
            "t_star_used": t_star,
            "t_patch_used": t_patch,
            "bifurcation_js": bif_js,
            "bifurcation_residual": bif_res,
            "corr_js_divergence_jetons": div["corr_js_divergence_jetons"],
            "mean_entropy": float(np.mean(div["entropy_mean"])),
            **self.summarize_patches(patch_results),
        }

        return BifurcationResult(
            prompt=prompt,
            n_trajectories=n_trajectories,
            residual_dist_over_time=div["residual_dist_over_time"],
            js_over_time=div["js_over_time"],
            bifurcation_times={
                "js": bif_js.get("t_star"),
                "residual": bif_res.get("t_star"),
            },
            patch_results=patch_results,
            summary=summary,
            residual_cos_over_time=div["residual_cos_over_time"],
            frac_pairs_same_prefix=div["frac_pairs_same_prefix"],
        )

    def cartographier(
        self,
        prompts: Sequence[str],
        n: int = 192,
        n_steps: int = 32,
        layer: Optional[int] = None,
        t_valeur: Optional[int] = None,
        seed_base: Optional[int] = None,
        npc: int = 10,
        seed_analyse: int = 0,
        rejouer: bool = False,
    ) -> Dict[str, Any]:
        """Point d'entrée pour l'atlas : un ensemble par prompt, puis, sur la représentation R_h (moyenne des états de la
        couche `layer` après lecture des jetons générés ; défaut : couche ⌊2L/3⌋ − 1, soit hidden_states[⌊2L/3⌋] de B2) :
        régimes (test, calibration, puissance, verdict de B2), et pour chaque prompt multimodal la bifurcation par
        préfixe et la valeur ajoutée de l'état à t_valeur sur un lecteur de jetons (étiquettes = régimes de la suite après
        t_valeur, définis sur les jetons futurs pour ne pas avantager les états). Renvoie des données sérialisables.

        rejouer : R_h par une passe complète sur prompt + suite (rejouer_moyennes, exactement la représentation de B2)
        au lieu de la moyenne accumulée pendant la génération (mêmes positions, KV-cache : arrondis fp16 un peu
        différents). Les réglages qui fixent le résultat sont renvoyés avec la carte."""
        layer = (2 * self.num_layers) // 3 - 1 if layer is None else layer
        t_valeur = n_steps // 4 if t_valeur is None else t_valeur
        if not 1 <= t_valeur < n_steps - 1:
            raise ValueError("t_valeur doit laisser au moins un jeton avant et deux après (1 ≤ t_valeur < n_steps − 1)")
        ens = self.generer_ensemble(prompts, n, n_steps, seed_base=seed_base, layers=[layer], keep_steps=True,
                                    mean_layers=[] if rejouer else [layer], steps_to_keep=[t_valeur])
        if rejouer:
            R_all = self.rejouer_moyennes([ens.prompt_ids[p] for p in ens.prompt_of], ens.tokens, [layer])[layer]
        else:
            R_all = ens.hidden_mean[layer]
        R = [R_all[ens.lignes_du_prompt(p)] for p in range(len(prompts))]
        reg = analyser_regimes(R, seed=seed_analyse, npc=npc)
        fiches = []
        for p in range(len(prompts)):
            li = ens.lignes_du_prompt(p)
            f = {"prompt": prompts[p], "multimodal": bool(reg["multimodal"][p]), "dip": reg["dip"][p], "p": reg["p"][p],
                 "suites_distinctes": len({tuple(r) for r in ens.tokens[li]}) / len(li),
                 "entropie_moyenne": float(ens.entropy_tsallis[li].mean())}
            if f["multimodal"]:
                lab = etiquettes_regimes(R[p], seed_analyse, npc)
                f["bifurcation"] = bifurcation_prefixe(ens.tokens[li], lab, ens.entropy_tsallis[li], seed_analyse)
                fut = etiquettes_regimes(sac_de_jetons([r[t_valeur + 1:] for r in ens.tokens[li]]), seed_analyse, npc)
                f["valeur_ajoutee"] = valeur_ajoutee_sur_jetons(
                    ens.hidden_steps[layer][li, 0].astype(np.float32), [r[:t_valeur] for r in ens.tokens[li]], fut,
                    seed=seed_analyse)
            fiches.append(f)
        return {"couche": layer, "t_valeur": t_valeur, "n": n, "n_steps": n_steps, "npc": npc,
                "seed_base": int(ens.seed_base), "seed_analyse": seed_analyse,
                "representation": "rejouee" if rejouer else "generation",
                "regimes": {k: v for k, v in reg.items()
                                                                  if k in ("pi", "pi_ci", "n_multi", "nul", "puissance",
                                                                           "verdict")},
                "fiches": fiches}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Trajectory Bifurcation Engine")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Raconte une histoire courte.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--n-patch-pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", default="gumbel", choices=["gumbel", "multinomial"])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Qwen3 : enable_thinking=False dans le chat template",
    )
    parser.add_argument(
        "--patch-without-bifurcation",
        action="store_true",
        help="Patcher même si aucune bifurcation n'est détectée (t >= 1)",
    )
    args = parser.parse_args()

    engine = TrajectoryBifurcationEngine(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        sampler=args.sampler,
        temperature=args.temperature,
    )
    result = engine.analyze_prompt(
        args.prompt,
        n_trajectories=args.n,
        n_patch_pairs=args.n_patch_pairs,
        chat_template_kwargs=(
            {"enable_thinking": False} if args.no_thinking else None
        ),
        require_bifurcation=not args.patch_without_bifurcation,
    )
    print(json.dumps(result.summary, indent=2, ensure_ascii=False, default=str))
    for r in result.patch_results:
        kind = r.pair_detail
        fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"
        print(
            f"[{kind}] {r.source_traj_idx}→{r.target_traj_idx} "
            f"recovery={fmt(r.recovery)} "
            f"multi-pas={fmt(r.recovery_multistep)} "
            f"loglik={fmt(r.recovery_loglik)} "
            f"overlap={r.final_token_overlap:.2f}"
        )
```
