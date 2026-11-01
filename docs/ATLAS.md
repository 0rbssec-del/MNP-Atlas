# Atlas v0.3 : couche de compatibilité et atlas des résultats

Commit `0c02b90` (branche `ccr-0f9a5c2e-elasx1`). Construit sur le moteur Core v0.2.0 (voir `docs/ENGINE.md`) sans le
modifier. 22 tests (`tests/test_atlas_compat.py`, `tests/test_bifurcation_atlas.py`) ; 67 avec ceux du moteur.

## 1. Organisation

| Fichier | Rôle |
|---|---|
| `atlas/compat.py` | Seul point de contact avec le moteur : protocoles figés, accès aux méthodes internes, garde de version, lecture des résultats |
| `atlas/bifurcation_atlas.py` | Atlas v0.3 : stockage, interrogation, comparaison et sauvegarde des `BifurcationResult` |
| `atlas/__init__.py` | Exports du paquet |
| `bifurcation_atlas.py` (racine) | Raccourci d'import `from bifurcation_atlas import BifurcationAtlas` |

Dépendances : `atlas/bifurcation_atlas.py` → `atlas/compat.py` → `trajectory_bifurcation_core.py`. Si l'API du moteur
change, seul `compat.py` doit suivre.

## 2. `atlas/compat.py`

**Protocoles figés.** Chaque usage nomme ses réglages au lieu de dépendre des défauts du moteur (qui ont changé en
v0.2) :

| Protocole | Échantillonneur | T | top-p | Prompt | Seuils de recovery | Usage |
|---|---|---|---|---|---|---|
| `v015` | multinomial | 0,8 | 0,95 | chat | 1e-4 / 1e-3 nat | refaire `resultats_multi_prompt.json` |
| `akarlar` | multinomial | 0,7 | 1,0 | brut | — | réplication d'Akarlar |
| `sgz` | Gumbel | 0,6 | 1,0 | brut | défauts v0.2 | analyses au protocole v0.2 |

`construire_moteur(protocole, **kw)` refuse un argument contraire au protocole (changer de réglage, c'est changer de
protocole) et marque le moteur (`protocole_atlas`). `verifier_moteur()` refuse un moteur sans l'API v0.2 en listant ce
qui manque.

**Accès internes centralisés.** `capture_hooks`, `patch_hook`, `js_exacte`, `generateur`, `generer_trajectoire`
délèguent aux méthodes privées du moteur ; aucune expérience ne les appelle directement.

**Provenance et lecture.** `version_moteur()` lit « Core vX.Y.Z » en tête du moteur ; `reglages_moteur()` rassemble les
réglages qui déterminent les tirages. `type_detaille()` donne le type d'une paire ; pour un `PatchResult` picklé avant la
v0.2 (sans `pair_detail`), il le déduit de `last_token_same` au lieu de la valeur par défaut de la classe
(« informative_token »), qui serait fausse. `resume_paire()` et `paires_du_type()` servent aux JSON des expériences.

**Vérifié.** Les protocoles `v015` et `akarlar` refont les tirages d'une réimplémentation indépendante de
l'échantillonneur v0.1.5 ; relancés sur vrais modèles, un run publié de `multi_prompt.py` (24 paires, 8 témoins nuls)
et le pool d'un prompt de la réplication (46 textes) sont identiques aux JSON publiés.

## 3. `atlas/bifurcation_atlas.py`

Adapté de l'atlas v0.2 (écrit pour le core v0.1.4). API et valeurs par défaut inchangées : les chiffres existants ne
bougent pas.

- **Entrées et provenance.** `add(result, provenance)` ; `build_from_prompts(engine_ou_protocole, prompts, ...)`
  enregistre les réglages du moteur pour chaque entrée. `summary()` signale les protocoles mélangés ;
  `compare_entries()` renvoie `comparable` (modèle, échantillonneur, T, top-p, gabarit, longueur identiques).
- **`EntrySummary`.** Champs v0.2 (t*, moyenne du recovery au 1er pas, taux de succès, entropie, `has_bifurcation`) et
  mesures v0.2 du moteur : `corr_js_divergence_jetons`, `bifurcation_suit_jetons` (≥ 0,9), `t_patch`, nombres de paires
  « contexte » / « jeton », médianes de `recovery_multistep` et `recovery_loglik` par type.
- **Requêtes.** `get_by_recovery` et `rank_by_recovery` prennent `mesure` (`recovery`, `recovery_multistep`,
  `recovery_loglik`), `detail` (`informative`, `informative_context`, `informative_token`) et `agregat` (`mean` par
  défaut, `median`).
- **Navigation.** `to_navigation_features` : valeur absente = NaN (−1,0 est un recovery possible).
- **Sérialisation.** `save` écrit `atlas_meta.json` (métadonnées, résumés, provenance) et `atlas_data.pkl` ; `load` relit
  les atlas v0.2 (provenance absente → None). Le pickle exécute du code : ne charger que des atlas de confiance.
- **Compatibilité v0.1.x.** Corrélation C1 recalculée depuis `js_over_time` et `frac_pairs_same_prefix` ; type détaillé
  via `type_detaille`.

**À lire avec prudence.** `has_bifurcation` suit surtout la divergence des jetons, et `mean_recovery_informative`
(moyenne au 1er pas) est dominé par l'identité du dernier jeton ; préférer les médianes multi-pas par type. Essai réel
(Qwen3-0.6B-Base, 2 prompts) : recovery moyen 0,81 contre 0,40 en médiane multi-pas des paires « jeton ».

## 4. Utilisation

```python
from bifurcation_atlas import BifurcationAtlas
atlas = BifurcationAtlas.build_from_prompts(
    "sgz", ["La ville de Lyon est située au confluent", "The city of Boston is known for"],
    n_trajectories=8, n_patch_pairs=3,
    moteur_kwargs=dict(max_new_tokens=32, layers_to_track=[0, 7, 14, 21]),
    forced_horizon=8, require_bifurcation=False)
atlas.print_summary()
atlas.save("atlas_sgz")
classement = atlas.rank_by_recovery(mesure="recovery_multistep", detail="informative_token", agregat="median")
```

## 5. Limites

- `build_from_prompts` passe par `analyze_prompt` du moteur : peu de paires « contexte ».
- Le pickle dépend des classes du moteur ; un export JSON complet serait plus durable.

## 6. Code complet

### `atlas/compat.py`

```python
# atlas/compat.py
# Couche de compatibilité entre le code atlas (expériences, analyses) et le
# moteur trajectory_bifurcation_core, qui reste séparé et n'est pas modifié ici.
#
# - Protocoles figés : chaque expérience nomme ses réglages au lieu de dépendre
#   des défauts du moteur (changés en v0.2 : Gumbel, T = 0,6, sans top-p).
# - Accès centralisé aux méthodes internes du moteur dont l'atlas a besoin
#   (hooks, JS exacte, générateur, trajectoire unique) : si le moteur change,
#   seul ce fichier change.
# - Garde de version : un moteur sans l'API v0.2 est refusé avec un message clair.

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Optional, Sequence

import torch

import trajectory_bifurcation_core as _core
from trajectory_bifurcation_core import (
    BifurcationResult,
    PatchResult,
    Trajectory,
    TrajectoryBifurcationEngine,
)

PROTOCOLES: Dict[str, Dict[str, Any]] = {
    # experiments/resultats_multi_prompt.json, produit avec le moteur v0.1.5
    "v015": {
        "moteur": dict(sampler="multinomial", temperature=0.8, top_p=0.95, chat=True),
        "modele": "Qwen/Qwen3-0.6B",
        # Seuils des rapports de recovery de v0.1.5 (v0.2 : 0,05 nat et 0,5 nat)
        "seuils_recovery": (1e-4, 1e-3),
    },
    # Réplication d'Akarlar : leur échantillonneur (torch.multinomial), T = 0,7,
    # sans top-p, prompt brut
    "akarlar": {
        "moteur": dict(sampler="multinomial", temperature=0.7, top_p=1.0, chat=False),
        "modele": "Qwen/Qwen2.5-1.5B",
        "seuils_recovery": None,
    },
    # Protocole SGZ du moteur v0.2 : Gumbel à nombres aléatoires communs,
    # T = 0,6, sans top-p, prompt brut, jetons spéciaux interdits
    "sgz": {
        "moteur": dict(sampler="gumbel", temperature=0.6, top_p=1.0, chat=False, q_tsallis=1.45),
        "modele": "Qwen/Qwen3-0.6B-Base",
        "seuils_recovery": None,
    },
}

# API du moteur dont dépend l'atlas
PARAMETRES_REQUIS = ("sampler", "chat")
METHODES_REQUISES = (
    "_capture_hooks", "_patch_hook", "_js_rows", "_make_generator",
    "_generate_one_trajectory", "generate_ensemble", "compute_divergences",
    "detect_bifurcation", "choose_patch_time", "reversibility_analysis",
    "forced_patch_effect", "continue_from_prefix", "classify_pair",
    "classify_pair_detail",
)


def version_moteur() -> str:
    """Version écrite en tête du moteur (« Core vX.Y.Z »), ou « inconnue »."""
    try:
        with open(_core.__file__, encoding="utf-8") as f:
            tete = f.read(400)
    except OSError:
        return "inconnue"
    m = re.search(r"Core v(\d+(?:\.\d+)+)", tete)
    return m.group(1) if m else "inconnue"


def reglages_moteur(moteur) -> Dict[str, Any]:
    """Réglages qui déterminent les tirages (provenance d'une entrée d'atlas)."""
    return {
        "protocole": getattr(moteur, "protocole_atlas", None),
        "modele": moteur.model_name,
        "sampler": moteur.sampler,
        "temperature": moteur.temperature,
        "top_p": moteur.top_p,
        "chat": moteur.chat,
        "seed": moteur.seed,
        "max_new_tokens": moteur.max_new_tokens,
        "min_js_den": moteur.MIN_JS_DEN,
        "min_ll_den": moteur.MIN_LL_DEN,
        "core_version": version_moteur(),
    }


def verifier_moteur(cls=TrajectoryBifurcationEngine) -> None:
    params = set(inspect.signature(cls.__init__).parameters)
    manque = [p for p in PARAMETRES_REQUIS if p not in params]
    manque += [m for m in METHODES_REQUISES if not hasattr(cls, m)]
    if manque:
        raise RuntimeError(
            f"Moteur incompatible avec l'atlas (API v0.2 requise) ; manque : {', '.join(manque)}"
        )


def construire_moteur(protocole: str, **kw) -> TrajectoryBifurcationEngine:
    """Moteur réglé selon `protocole` ; les autres arguments passent au moteur.

    Un argument qui contredit le protocole est refusé : changer d'échantillonneur
    ou de température, c'est changer de protocole.
    """
    if protocole not in PROTOCOLES:
        raise ValueError(f"Protocole inconnu : {protocole} ({', '.join(PROTOCOLES)})")
    verifier_moteur()
    proto = PROTOCOLES[protocole]
    conflits = [k for k, v in proto["moteur"].items() if k in kw and kw[k] != v]
    if conflits:
        raise ValueError(f"Réglages contraires au protocole {protocole} : {', '.join(conflits)}")
    kw.setdefault("model_name", proto["modele"])
    moteur = TrajectoryBifurcationEngine(**{**kw, **proto["moteur"]})
    if proto["seuils_recovery"] is not None:
        moteur.MIN_JS_DEN, moteur.MIN_LL_DEN = proto["seuils_recovery"]
    moteur.protocole_atlas = protocole
    return moteur


# ----------------------------------------------------------------------
# Méthodes internes du moteur utilisées par l'atlas
# ----------------------------------------------------------------------

def capture_hooks(moteur, couches: Sequence[int]):
    """Contexte : sortie brute de chaque couche à la dernière position."""
    return moteur._capture_hooks(list(couches))


def patch_hook(moteur, couche: int, position: int, vecteur: torch.Tensor):
    """Contexte : remplace la sortie de `couche` à `position` (premier passage)."""
    return moteur._patch_hook(couche, position, vecteur)


def js_exacte(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """JS exacte (nats) ligne par ligne entre distributions complètes."""
    return TrajectoryBifurcationEngine._js_rows(p, q)


def generateur(moteur, graine: Optional[int]) -> Optional[torch.Generator]:
    return moteur._make_generator(graine)


def generer_trajectoire(moteur, prompt_ids: torch.Tensor, graine: Optional[int]) -> Trajectory:
    """Une trajectoire (API v0.1.x du moteur), avec l'échantillonneur du protocole."""
    return moteur._generate_one_trajectory(prompt_ids, seed=graine)


# ----------------------------------------------------------------------
# Lecture des résultats du moteur
# ----------------------------------------------------------------------

def resume_paire(r: PatchResult) -> Dict[str, Any]:
    """Champs d'un PatchResult gardés dans les JSON de l'atlas."""
    return {
        "kind": r.pair_kind,
        "detail": r.pair_detail,
        "src": r.source_traj_idx,
        "tgt": r.target_traj_idx,
        "first_diff_step": r.first_diff_step,
        "last_token_same": r.last_token_same,
        "recovery": r.recovery,
        "recovery_step0": (r.recovery_per_step or [None])[0],
        "recovery_multistep": r.recovery_multistep,
        "recovery_loglik": r.recovery_loglik,
        "overlap": r.final_token_overlap,
    }


def type_detaille(p: PatchResult) -> str:
    """Type détaillé d'une paire : témoins, informative_context ou informative_token.

    Un PatchResult picklé avant la v0.2 n'a pas pair_detail (la valeur par défaut
    de la classe, « informative_token », serait fausse) : on le déduit de
    last_token_same, ou on renvoie « informative » si ce champ manque aussi.
    """
    if p.pair_kind != "informative":
        return p.pair_kind
    champs = vars(p)
    if champs.get("pair_detail") is not None:
        return champs["pair_detail"]
    same = champs.get("last_token_same")
    if same is None:
        return "informative"
    return "informative_context" if same else "informative_token"


def paires_du_type(paires: List[Dict[str, Any]], detail: str) -> List[Dict[str, Any]]:
    """Paires d'un type détaillé (informative_context, informative_token…).

    Les JSON écrits avant la v0.2 n'ont pas « detail » : on le déduit de
    last_token_same, qui porte la même information.
    """
    def det(p):
        if p.get("detail") is not None or p["kind"] != "informative":
            return p.get("detail", p["kind"])
        return "informative_context" if p.get("last_token_same") else "informative_token"
    return [p for p in paires if det(p) == detail]


__all__ = [
    "PROTOCOLES", "BifurcationResult", "PatchResult", "Trajectory", "TrajectoryBifurcationEngine",
    "version_moteur", "reglages_moteur", "verifier_moteur", "construire_moteur",
    "capture_hooks", "patch_hook", "js_exacte", "generateur", "generer_trajectoire",
    "resume_paire", "type_detaille", "paires_du_type",
]
```

### `atlas/bifurcation_atlas.py`

```python
# atlas/bifurcation_atlas.py
# Atlas v0.3 – Stockage, interrogation et comparaison des BifurcationResult
# Compatible Core v0.2 (via atlas/compat.py) ; relit les atlas écrits avec les
# cores v0.1.4 et v0.1.5.
#
# Changements v0.2 → v0.3 :
#   - Plus d'accès direct au moteur : types, construction, version et
#     réglages passent par atlas/compat.py (le moteur reste séparé).
#   - Provenance par entrée (protocole, échantillonneur, T, top-p, modèle…) :
#     deux entrées de protocoles différents ne sont pas comparables.
#   - Mesures de v0.2 dans les résumés : corrélation JS / divergence des
#     jetons (≥ 0,9 : la « bifurcation » JS suit les jetons), médianes de
#     recovery multi-pas et log-vraisemblance par type de paire (« contexte »,
#     dernier jeton identique ; « jeton », dernier jeton différent). Les champs
#     de v0.2 gardent leur sens (moyenne du recovery au 1er pas).
#   - Filtres et classements : choix de la mesure, du type de paire et de
#     l'agrégat (moyenne par défaut, comme v0.2 ; la médiane résiste mieux aux
#     rapports instables).
#   - to_navigation_features : NaN au lieu de -1.0 pour une valeur absente
#     (-1.0 est un recovery possible).
#   - Horodatage sans datetime.utcnow (obsolète depuis Python 3.12).

from __future__ import annotations
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np

from atlas.compat import (
    BifurcationResult,
    PatchResult,
    TrajectoryBifurcationEngine,
    construire_moteur,
    reglages_moteur,
    type_detaille,
    version_moteur,
)

ATLAS_VERSION = "0.3"
# Critère C1 : au-delà, la « bifurcation » JS suit la divergence des jetons
SEUIL_SUIT_JETONS = 0.9
MESURES_RECOVERY = ("recovery", "recovery_multistep", "recovery_loglik")
# Réglages qui doivent coïncider pour comparer deux entrées
CLES_COMPARABLES = ("modele", "sampler", "temperature", "top_p", "chat", "max_new_tokens")


@dataclass
class EntrySummary:
    """Résumé compact et sérialisable d’une entrée."""
    prompt: str
    n_trajectories: int
    t_star_js: Optional[int]
    t_star_residual: Optional[int]
    # Moyenne du recovery au 1er pas (top-k) sur les paires informatives
    mean_recovery_informative: Optional[float]
    n_informative_patches: int
    patch_success_rate: Optional[float]
    mean_entropy: Optional[float]
    has_bifurcation: bool
    # Mesures de v0.2
    corr_js_divergence_jetons: Optional[float] = None
    bifurcation_suit_jetons: Optional[bool] = None
    t_patch: Optional[int] = None
    n_informative_context: int = 0
    n_informative_token: int = 0
    median_recovery_multistep_context: Optional[float] = None
    median_recovery_multistep_token: Optional[float] = None
    median_recovery_loglik_context: Optional[float] = None
    median_recovery_loglik_token: Optional[float] = None


def _agreger(values, agregat: str = "mean") -> Optional[float]:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return float(np.median(v) if agregat == "median" else np.mean(v))


def _paires(result: BifurcationResult, detail: str = "informative") -> List[PatchResult]:
    """Paires informatives (« informative ») ou d'un type détaillé."""
    if detail == "informative":
        return [p for p in result.patch_results if p.pair_kind == "informative"]
    return [p for p in result.patch_results if type_detaille(p) == detail]


def _corr_jetons(result: BifurcationResult) -> Optional[float]:
    """Corrélation JS(t) / part des paires divergentes ; recalculée pour une
    entrée v0.1.x (qui a js_over_time et frac_pairs_same_prefix)."""
    c = result.summary.get("corr_js_divergence_jetons")
    if c is not None:
        return float(c)
    if result.js_over_time is None or result.frac_pairs_same_prefix is None:
        return None
    js = np.asarray(result.js_over_time, float)
    frac = np.asarray(result.frac_pairs_same_prefix, float)
    if len(js) < 3 or np.std(js) == 0 or np.std(frac) == 0:
        return None
    return float(np.corrcoef(js, 1 - frac)[0, 1])


def _score_recovery(
    result: BifurcationResult,
    mesure: str = "recovery",
    detail: str = "informative",
    agregat: str = "mean",
    min_pairs: int = 1,
) -> Optional[float]:
    if mesure not in MESURES_RECOVERY:
        raise ValueError(f"mesure : {', '.join(MESURES_RECOVERY)}")
    vals = [getattr(p, mesure, None) for p in _paires(result, detail)]
    vals = [v for v in vals if v is not None]
    if len(vals) < min_pairs:
        return None
    return _agreger(vals, agregat)


def _nan(x) -> float:
    return float("nan") if x is None else float(x)


class BifurcationAtlas:
    def __init__(self, name: str = "default_atlas"):
        self.name = name
        self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.entries: List[BifurcationResult] = []
        # Réglages du moteur de chaque entrée (None si inconnus)
        self.provenance: List[Optional[Dict[str, Any]]] = []
        self.metadata: Dict[str, Any] = {
            "name": name,
            "created_at": self.created_at,
            "core_version": version_moteur(),
            "atlas_version": ATLAS_VERSION,
            "n_entries": 0,
        }
        # Index léger : prompt → dernière entrée de ce prompt
        self._prompt_index: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Gestion des entrées
    # ------------------------------------------------------------------

    def add(self, result: BifurcationResult, provenance: Optional[Dict[str, Any]] = None) -> int:
        """Ajoute une entrée et retourne son index. `provenance` : réglages du
        moteur qui l'a produite (atlas.compat.reglages_moteur)."""
        idx = len(self.entries)
        self.entries.append(result)
        self.provenance.append(provenance)
        self._prompt_index[result.prompt] = idx
        self.metadata["n_entries"] = len(self.entries)
        return idx

    def add_many(self, results: List[BifurcationResult], provenance: Optional[Dict[str, Any]] = None) -> None:
        for r in results:
            self.add(r, provenance)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> BifurcationResult:
        return self.entries[idx]

    # ------------------------------------------------------------------
    # Résumé d’une entrée
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_entry(result: BifurcationResult) -> EntrySummary:
        info_patches = _paires(result)
        successes = [p.success for p in info_patches]
        success_rate = float(np.mean(successes)) if successes else None
        mean_ent = None
        if "mean_entropy" in result.summary:
            mean_ent = result.summary["mean_entropy"]
        elif "entropy_mean" in result.summary:
            # compatibilité éventuelle
            ent = result.summary["entropy_mean"]
            mean_ent = float(np.mean(ent)) if isinstance(ent, (list, np.ndarray)) else ent

        t_js = result.bifurcation_times.get("js")
        t_res = result.bifurcation_times.get("residual")
        corr = _corr_jetons(result)
        ctx = _paires(result, "informative_context")
        tok = _paires(result, "informative_token")

        def med(paires, mesure):
            return _agreger([getattr(p, mesure, None) for p in paires], "median")

        return EntrySummary(
            prompt=result.prompt,
            n_trajectories=result.n_trajectories,
            t_star_js=t_js,
            t_star_residual=t_res,
            mean_recovery_informative=_score_recovery(result),
            n_informative_patches=len(info_patches),
            patch_success_rate=success_rate,
            mean_entropy=mean_ent,
            has_bifurcation=(t_js is not None) or (t_res is not None),
            corr_js_divergence_jetons=corr,
            bifurcation_suit_jetons=None if corr is None else corr >= SEUIL_SUIT_JETONS,
            t_patch=result.summary.get("t_patch_used"),
            n_informative_context=len(ctx),
            n_informative_token=len(tok),
            median_recovery_multistep_context=med(ctx, "recovery_multistep"),
            median_recovery_multistep_token=med(tok, "recovery_multistep"),
            median_recovery_loglik_context=med(ctx, "recovery_loglik"),
            median_recovery_loglik_token=med(tok, "recovery_loglik"),
        )

    # ------------------------------------------------------------------
    # Requêtes
    # ------------------------------------------------------------------

    def filter(
        self, predicate: Callable[[BifurcationResult], bool]
    ) -> List[BifurcationResult]:
        return [e for e in self.entries if predicate(e)]

    def get_by_prompt(self, prompt: str) -> Optional[BifurcationResult]:
        """Dernière entrée de ce prompt (voir get_all_by_prompt)."""
        idx = self._prompt_index.get(prompt)
        return self.entries[idx] if idx is not None else None

    def get_all_by_prompt(self, prompt: str) -> List[BifurcationResult]:
        return self.filter(lambda e: e.prompt == prompt)

    def get_by_prompt_substring(self, text: str) -> List[BifurcationResult]:
        text = text.lower()
        return self.filter(lambda e: text in e.prompt.lower())

    def get_with_bifurcation(
        self, measure: str = "any"
    ) -> List[BifurcationResult]:
        if measure == "js":
            return self.filter(
                lambda e: e.bifurcation_times.get("js") is not None
            )
        if measure == "residual":
            return self.filter(
                lambda e: e.bifurcation_times.get("residual") is not None
            )
        # any
        return self.filter(
            lambda e: (
                e.bifurcation_times.get("js") is not None
                or e.bifurcation_times.get("residual") is not None
            )
        )

    def get_by_recovery(
        self,
        min_recovery: Optional[float] = None,
        max_recovery: Optional[float] = None,
        min_pairs: int = 2,
        mesure: str = "recovery",
        detail: str = "informative",
        agregat: str = "mean",
    ) -> List[BifurcationResult]:
        def pred(e: BifurcationResult) -> bool:
            score = _score_recovery(e, mesure, detail, agregat, min_pairs)
            if score is None:
                return False
            if min_recovery is not None and score < min_recovery:
                return False
            if max_recovery is not None and score > max_recovery:
                return False
            return True

        return self.filter(pred)

    def rank_by_recovery(
        self,
        descending: bool = True,
        mesure: str = "recovery",
        detail: str = "informative",
        agregat: str = "mean",
    ) -> List[Tuple[float, BifurcationResult]]:
        scored = []
        for e in self.entries:
            score = _score_recovery(e, mesure, detail, agregat)
            if score is not None:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=descending)
        return scored

    def rank_by_bifurcation_time(
        self, measure: str = "js", earliest_first: bool = True
    ) -> List[Tuple[int, BifurcationResult]]:
        scored = []
        for e in self.entries:
            t = e.bifurcation_times.get(measure)
            if t is not None:
                scored.append((t, e))
        scored.sort(key=lambda x: x[0], reverse=not earliest_first)
        return scored

    # ------------------------------------------------------------------
    # Résumés globaux
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        if not self.entries:
            return {"n_entries": 0, "atlas_version": ATLAS_VERSION}

        n = len(self.entries)
        entry_summaries = [self.summarize_entry(e) for e in self.entries]

        has_bif = sum(1 for s in entry_summaries if s.has_bifurcation)
        has_js = sum(1 for s in entry_summaries if s.t_star_js is not None)
        has_res = sum(1 for s in entry_summaries if s.t_star_residual is not None)

        recoveries = [
            s.mean_recovery_informative
            for s in entry_summaries
            if s.mean_recovery_informative is not None
        ]
        bif_times_js = [
            s.t_star_js for s in entry_summaries if s.t_star_js is not None
        ]
        corrs = [s.bifurcation_suit_jetons for s in entry_summaries
                 if s.bifurcation_suit_jetons is not None]

        def pooled(detail, mesure):
            return _agreger(
                [getattr(p, mesure, None) for e in self.entries for p in _paires(e, detail)],
                "median",
            )

        protocoles = Counter(
            (p or {}).get("protocole") or "inconnu" for p in self.provenance
        )
        return {
            "n_entries": n,
            "atlas_version": ATLAS_VERSION,
            "fraction_with_any_bifurcation": has_bif / n,
            "fraction_with_bifurcation_js": has_js / n,
            "fraction_with_bifurcation_residual": has_res / n,
            "mean_recovery_informative": (
                float(np.mean(recoveries)) if recoveries else None
            ),
            "std_recovery_informative": (
                float(np.std(recoveries)) if recoveries else None
            ),
            "median_t_star_js": (
                float(np.median(bif_times_js)) if bif_times_js else None
            ),
            "n_entries_with_recovery": len(recoveries),
            # Mesures de v0.2
            "fraction_bifurcation_suit_jetons": (
                sum(corrs) / len(corrs) if corrs else None
            ),
            "median_recovery_multistep_context": pooled("informative_context", "recovery_multistep"),
            "median_recovery_multistep_token": pooled("informative_token", "recovery_multistep"),
            "median_recovery_loglik_context": pooled("informative_context", "recovery_loglik"),
            "median_recovery_loglik_token": pooled("informative_token", "recovery_loglik"),
            "protocoles": dict(protocoles),
            "protocoles_melanges": len(protocoles) > 1,
        }

    def print_summary(self, top_k: int = 5) -> None:
        s = self.summary()
        print(f"=== Atlas « {self.name} » (v{s.get('atlas_version', '?')}) ===")
        print(f"Entrées                     : {s['n_entries']}")
        if not s["n_entries"]:
            return
        print(f"Avec bifurcation (any)      : {s['fraction_with_any_bifurcation']:.1%}")
        print(f"Avec bifurcation JS         : {s['fraction_with_bifurcation_js']:.1%}")
        print(f"Avec bifurcation Residual   : {s['fraction_with_bifurcation_residual']:.1%}")
        if s["fraction_bifurcation_suit_jetons"] is not None:
            print(f"JS suit les jetons (≥ {SEUIL_SUIT_JETONS}) : {s['fraction_bifurcation_suit_jetons']:.1%}")
        if s["mean_recovery_informative"] is not None:
            print(
                f"Recovery moyen (informatif) : "
                f"{s['mean_recovery_informative']:.3f} ± {s['std_recovery_informative']:.3f}"
            )
        for d in ("context", "token"):
            m = s[f"median_recovery_multistep_{d}"]
            if m is not None:
                print(f"Recovery multi-pas médian ({d}) : {m:.3f}")
        if s["median_t_star_js"] is not None:
            print(f"Médiane t* (JS)             : {s['median_t_star_js']:.1f}")
        print(f"Protocoles                  : {s['protocoles']}")
        if s["protocoles_melanges"]:
            print("ATTENTION : protocoles mélangés, entrées non comparables entre elles")

        # Top / Bottom recovery
        ranked = self.rank_by_recovery(descending=True)
        if ranked:
            print(f"\nTop {min(top_k, len(ranked))} recovery les plus élevés :")
            for rec, e in ranked[:top_k]:
                print(f"  {rec:.3f}  |  {e.prompt[:70]}")

            print(f"\nTop {min(top_k, len(ranked))} recovery les plus bas :")
            for rec, e in ranked[-top_k:][::-1]:
                print(f"  {rec:.3f}  |  {e.prompt[:70]}")

    # ------------------------------------------------------------------
    # Comparaison entre entrées
    # ------------------------------------------------------------------

    def compare_entries(
        self, idx_a: int, idx_b: int
    ) -> Dict[str, Any]:
        """Compare deux entrées de façon factuelle."""
        a = self.entries[idx_a]
        b = self.entries[idx_b]
        sa = self.summarize_entry(a)
        sb = self.summarize_entry(b)
        pa, pb = self.provenance[idx_a], self.provenance[idx_b]
        comparable = None
        if pa is not None and pb is not None:
            comparable = all(pa.get(k) == pb.get(k) for k in CLES_COMPARABLES)

        def ab(champ):
            return {"a": getattr(sa, champ), "b": getattr(sb, champ)}

        return {
            "prompt_a": sa.prompt[:80],
            "prompt_b": sb.prompt[:80],
            "t_star_js": ab("t_star_js"),
            "t_star_residual": ab("t_star_residual"),
            "mean_recovery": ab("mean_recovery_informative"),
            "has_bifurcation": ab("has_bifurcation"),
            "n_informative_patches": ab("n_informative_patches"),
            "corr_js_divergence_jetons": ab("corr_js_divergence_jetons"),
            "median_recovery_multistep_context": ab("median_recovery_multistep_context"),
            "median_recovery_multistep_token": ab("median_recovery_multistep_token"),
            # None : provenance inconnue pour au moins une entrée
            "comparable": comparable,
        }

    # ------------------------------------------------------------------
    # Features pour future Navigation Layer
    # ------------------------------------------------------------------

    def to_navigation_features(
        self, result: BifurcationResult
    ) -> Dict[str, float]:
        """
        Extrait un vecteur de features simple et interprétable
        pour une future projection / carte de navigation.
        Aucune promesse géométrique. Valeur absente : NaN.
        """
        s = self.summarize_entry(result)
        return {
            "t_star_js": _nan(s.t_star_js),
            "t_star_residual": _nan(s.t_star_residual),
            "mean_recovery": _nan(s.mean_recovery_informative),
            "patch_success_rate": _nan(s.patch_success_rate),
            "mean_entropy": _nan(s.mean_entropy),
            "has_bifurcation": 1.0 if s.has_bifurcation else 0.0,
            "corr_js_divergence_jetons": _nan(s.corr_js_divergence_jetons),
            "median_recovery_multistep_context": _nan(s.median_recovery_multistep_context),
            "median_recovery_multistep_token": _nan(s.median_recovery_multistep_token),
        }

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def save(self, directory: str, overwrite: bool = False) -> None:
        path = Path(directory)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} existe déjà. Utilise overwrite=True.")
        path.mkdir(parents=True, exist_ok=True)

        # Métadonnées + résumé global
        meta = {
            **self.metadata,
            "summary": self.summary(),
            "entry_summaries": [
                asdict(self.summarize_entry(e)) for e in self.entries
            ],
            "provenance": self.provenance,
        }
        with open(path / "atlas_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

        # Données complètes (pickle : ne charger que des atlas de confiance)
        with open(path / "atlas_data.pkl", "wb") as f:
            pickle.dump(self.entries, f)

        print(f"[Atlas] Sauvegardé → {path}")

    @classmethod
    def load(cls, directory: str) -> "BifurcationAtlas":
        """Charge un atlas v0.2 ou v0.3. Le pickle exécute du code : ne charger
        que des atlas de confiance."""
        path = Path(directory)
        with open(path / "atlas_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        with open(path / "atlas_data.pkl", "rb") as f:
            entries = pickle.load(f)

        atlas = cls(name=meta.get("name", "loaded_atlas"))
        atlas.created_at = meta.get("created_at", atlas.created_at)
        atlas.metadata = {**meta, "core_version_chargement": version_moteur()}
        atlas.entries = entries
        # Atlas v0.2 : pas de provenance enregistrée
        atlas.provenance = meta.get("provenance") or [None] * len(entries)
        atlas._rebuild_index()
        return atlas

    def _rebuild_index(self) -> None:
        self._prompt_index = {
            e.prompt: i for i, e in enumerate(self.entries)
        }

    # ------------------------------------------------------------------
    # Construction en lot
    # ------------------------------------------------------------------

    @classmethod
    def build_from_prompts(
        cls,
        engine: Union[TrajectoryBifurcationEngine, str],
        prompts: List[str],
        n_trajectories: int = 10,
        run_patching: bool = True,
        n_patch_pairs: int = 4,
        name: str = "batch_atlas",
        moteur_kwargs: Optional[Dict[str, Any]] = None,
        **analyze_kwargs,
    ) -> "BifurcationAtlas":
        """`engine` : un moteur déjà construit, ou le nom d'un protocole de
        atlas.compat (« sgz », « v015 »…), construit avec `moteur_kwargs`.
        `analyze_kwargs` passe à analyze_prompt (v0.2 : forced_horizon,
        require_bifurcation, chat_template_kwargs…)."""
        if isinstance(engine, str):
            engine = construire_moteur(engine, **(moteur_kwargs or {}))
        provenance = {
            **reglages_moteur(engine),
            "n_trajectories": n_trajectories,
            "n_patch_pairs": n_patch_pairs,
            "analyze_kwargs": analyze_kwargs,
        }
        atlas = cls(name=name)
        for i, prompt in enumerate(prompts):
            print(f"\n=== Prompt {i+1}/{len(prompts)} ===")
            result = engine.analyze_prompt(
                prompt=prompt,
                n_trajectories=n_trajectories,
                run_patching=run_patching,
                n_patch_pairs=n_patch_pairs,
                **analyze_kwargs,
            )
            atlas.add(result, provenance)
        return atlas
```

### `atlas/__init__.py`

```python
# Code atlas : expériences et analyses construites sur le moteur
# trajectory_bifurcation_core, sans le modifier (voir atlas/README.md).

from atlas.compat import PROTOCOLES, construire_moteur, verifier_moteur

__all__ = ["PROTOCOLES", "construire_moteur", "verifier_moteur"]
```

### `bifurcation_atlas.py`

```python
# bifurcation_atlas.py
# Raccourci d'import : l'atlas est dans atlas/bifurcation_atlas.py, séparé du
# moteur (trajectory_bifurcation_core.py).

from atlas.bifurcation_atlas import ATLAS_VERSION, BifurcationAtlas, EntrySummary

__all__ = ["ATLAS_VERSION", "BifurcationAtlas", "EntrySummary"]
```
