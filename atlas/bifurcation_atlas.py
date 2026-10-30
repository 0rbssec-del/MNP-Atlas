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
