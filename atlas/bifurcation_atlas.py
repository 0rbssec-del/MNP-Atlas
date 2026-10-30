# atlas/bifurcation_atlas.py
# Atlas v0.4 – Cartes de l'espace caché, et stockage des BifurcationResult
# Compatible Core v0.2.1 (via atlas/compat.py) ; relit les atlas v0.2 et v0.3.
#
# Changements v0.3 → v0.4 :
#   - Cartes de l'espace caché (type d'entrée principal) : sortie de
#     cartographier du moteur, une fiche par prompt (multimodal ou non, dip,
#     bifurcation par préfixe, valeur ajoutée de l'état sur un lecteur de
#     jetons), régimes de B2 pour l'ensemble. Stockées en JSON.
#   - Requêtes sur les cartes : fiches filtrées, signature d'un prompt,
#     matrice de signatures, accord entre deux cartes (prompts multimodaux
#     partagés, p hypergéométrique), résumé par carte.
#   - Comparabilité à trois états (oui / non / inconnu) sur des réglages
#     complets : précision de calcul, jetons spéciaux, seuils de recovery,
#     arguments d'analyse (réflexion on/off)… Un réglage absent d'une
#     provenance donne « inconnu », plus « oui ».
#   - Les mesures v0.1 (t* JS, recovery au 1er pas, has_bifurcation) sont
#     gardées telles quelles mais présentées comme historiques : elles suivent
#     la divergence des jetons et l'identité du dernier jeton (MOTEUR_V02.md §3).
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

from math import comb

from atlas import compat
from atlas.compat import (
    BifurcationResult,
    PatchResult,
    TrajectoryBifurcationEngine,
    construire_moteur,
    reglages_moteur,
    type_detaille,
    version_moteur,
)

ATLAS_VERSION = "0.4"
# Critère C1 : au-delà, la « bifurcation » JS suit la divergence des jetons
SEUIL_SUIT_JETONS = 0.9
MESURES_RECOVERY = ("recovery", "recovery_multistep", "recovery_loglik")
# Réglages qui doivent coïncider pour comparer deux entrées historiques (BifurcationResult)
CLES_COMPARABLES = ("modele", "sampler", "temperature", "top_p", "chat", "max_new_tokens", "forbid_special", "dtype",
                    "min_js_den", "min_ll_den", "topk_for_divergence", "analyze_kwargs")
# … et deux cartes (le nombre de pas remplace max_new_tokens ; seeds exclues : autres tirages, même expérience)
CLES_CARTES = ("modele", "sampler", "temperature", "top_p", "chat", "forbid_special", "q_tsallis", "dtype",
               "couche", "t_valeur", "n", "n_steps", "npc", "representation")
# Champs d'une signature de prompt (valeur absente : NaN)
CHAMPS_SIGNATURE = ("multimodal", "dip", "p", "suites_distinctes", "entropie_moyenne", "prefixe_k", "prefixe_gain_max",
                    "valeur_ecart", "valeur_ecart_bas", "valeur_ajoutee")


def comparer_provenances(pa: Optional[Dict[str, Any]], pb: Optional[Dict[str, Any]], cles: Tuple[str, ...],
                         ignorer: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """comparable : True si tous les réglages coïncident, False si l'un diffère, None si l'un manque (provenance
    inconnue ou écrite par une version qui ne l'enregistrait pas) sans qu'aucun ne diffère."""
    cles = tuple(k for k in cles if k not in ignorer)
    if pa is None or pb is None:
        return {"comparable": None, "differences": [], "inconnus": list(cles)}
    diff = [k for k in cles if k in pa and k in pb and pa[k] != pb[k]]
    inc = [k for k in cles if k not in pa or k not in pb]
    return {"comparable": False if diff else (None if inc else True), "differences": diff, "inconnus": inc}


def _json(x):
    """Copie sérialisable en JSON (types numpy, tuples) ; NaN gardé (lu par json de Python)."""
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def p_hypergeometrique(n: int, a: int, b: int, k: int) -> float:
    """P(X ≥ k) quand b prompts sont tirés au hasard parmi n dont a marqués : probabilité d'au moins k prompts
    multimodaux partagés par hasard."""
    if k <= 0:
        return 1.0
    return sum(comb(a, x) * comb(n - a, b - x) for x in range(k, min(a, b) + 1)) / comb(n, b)


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
        # Cartes de l'espace caché (v0.4) : {"nom", "cree", "carte", "provenance"}
        self.cartes: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "name": name,
            "created_at": self.created_at,
            "core_version": version_moteur(),
            "atlas_version": ATLAS_VERSION,
            "n_entries": 0,
            "n_cartes": 0,
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
        cartes = {"n_cartes": len(self.cartes), "cartes": [self.resume_carte(i) for i in range(len(self.cartes))]}
        if not self.entries:
            return {"n_entries": 0, "atlas_version": ATLAS_VERSION, **cartes}

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
            **cartes,
        }

    def print_summary(self, top_k: int = 5) -> None:
        s = self.summary()
        print(f"=== Atlas « {self.name} » (v{s.get('atlas_version', '?')}) ===")
        print(f"Cartes de l'espace caché    : {s['n_cartes']}")
        for c in s["cartes"]:
            r = c["regimes"]
            print(f"  {c['nom']} (couche {c['couche']}, R_h {c['representation']}) : {c['n_multimodaux']}/{c['n_prompts']} "
                  f"prompts multimodaux (π = {r['pi']:.2f}, IC95 [{r['pi_ci'][0]:.2f} ; {r['pi_ci'][1]:.2f}]) ; "
                  f"substitut gaussien {r['nul']:.0%}, mélanges {r['puissance']:.0%} → {r['verdict']}")
            if c["n_multimodaux"]:
                g = c["prefixe_gain_max_median"]
                print(f"    valeur ajoutée des états sur les jetons : {c['valeur_ajoutee']} ; préfixe décisif "
                      f"{c['prefixe_k_trouve']}/{c['n_multimodaux']}, gain médian {'n/a' if g is None else f'{g:.2f}'}")
        for i in range(len(self.cartes)):
            for j in range(i + 1, len(self.cartes)):
                a = self.accord_cartes(i, j)
                if a["n_communs"]:
                    print(f"  accord {self.cartes[i]['nom']} / {self.cartes[j]['nom']} : {a['partages']} prompts multimodaux "
                          f"partagés sur {a['n_communs']} (attendu {a['attendu']:.1f}, p = {a['p_hypergeometrique']:.3g})")
        print(f"Entrées historiques (v0.1)  : {s['n_entries']}")
        if not s["n_entries"]:
            return
        print("  t* JS, has_bifurcation et recovery au 1er pas suivent la divergence des jetons et l'identité du dernier")
        print("  jeton (MOTEUR_V02.md §3) : ce ne sont pas des mesures de l'espace caché.")
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
        comp = comparer_provenances(self.provenance[idx_a], self.provenance[idx_b], CLES_COMPARABLES)

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
            # True / False ; None : un réglage inconnu pour au moins une entrée (voir « inconnus »)
            "comparable": comp["comparable"],
            "differences": comp["differences"],
            "inconnus": comp["inconnus"],
        }

    # ------------------------------------------------------------------
    # Cartes de l'espace caché (v0.4)
    # ------------------------------------------------------------------

    def add_carte(self, carte: Dict[str, Any], provenance: Optional[Dict[str, Any]] = None,
                  nom: Optional[str] = None) -> int:
        """Ajoute une carte (sortie de cartographier du moteur) et retourne son index. `provenance` : réglages du
        moteur (atlas.compat.reglages_moteur)."""
        if not isinstance(carte, dict) or "fiches" not in carte or "regimes" not in carte:
            raise ValueError("carte : dictionnaire de cartographier attendu (clés « regimes » et « fiches »)")
        prov = None if provenance is None else _json(provenance)
        self.cartes.append({
            "nom": nom or (prov or {}).get("modele") or f"carte_{len(self.cartes)}",
            "cree": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "carte": _json(carte),
            "provenance": prov,
        })
        self.metadata["n_cartes"] = len(self.cartes)
        return len(self.cartes) - 1

    def cartographier(self, engine: Union[TrajectoryBifurcationEngine, str], prompts: List[str],
                      nom: Optional[str] = None, moteur_kwargs: Optional[Dict[str, Any]] = None, **kw) -> int:
        """Construit une carte avec le moteur (ou un protocole de atlas.compat construit avec `moteur_kwargs`) et
        l'ajoute. `kw` : arguments de cartographier (n, n_steps, layer, t_valeur, seed_base, npc, seed_analyse,
        rejouer)."""
        if isinstance(engine, str):
            engine = construire_moteur(engine, **(moteur_kwargs or {}))
        carte = compat.cartographier(engine, prompts, **kw)
        return self.add_carte(carte, reglages_moteur(engine), nom)

    def _carte(self, i: int) -> Dict[str, Any]:
        return self.cartes[i]["carte"]

    def parametres_carte(self, i: int) -> Dict[str, Any]:
        """Provenance du moteur et réglages de la carte (couche, t_valeur, n, n_steps, npc, représentation)."""
        c = self._carte(i)
        par = {k: c[k] for k in ("couche", "t_valeur", "n", "n_steps", "npc", "seed_base", "seed_analyse",
                                 "representation") if k in c}
        return {**(self.cartes[i]["provenance"] or {}), **par}

    def fiches(self, i: Optional[int] = None, multimodal: Optional[bool] = None,
               verdict_valeur: Optional[str] = None) -> List[Tuple[int, Dict[str, Any]]]:
        """(index de carte, fiche) ; filtres : carte i, prompts multimodaux ou non, verdict de la valeur ajoutée."""
        out = []
        for j in (range(len(self.cartes)) if i is None else [i]):
            for f in self._carte(j)["fiches"]:
                if multimodal is not None and bool(f.get("multimodal")) != multimodal:
                    continue
                if verdict_valeur is not None and (f.get("valeur_ajoutee") or {}).get("verdict") != verdict_valeur:
                    continue
                out.append((j, f))
        return out

    def signature(self, i: int, prompt: str) -> Dict[str, float]:
        """Signature d'un prompt dans la carte i : ce que la carte dit de son espace caché. Valeur absente : NaN
        (prompt unimodal : pas de bifurcation par préfixe ni de valeur ajoutée). valeur_ajoutee : 1 si VALEUR AJOUTÉE,
        0 si un autre verdict, NaN si non testée."""
        f = next((f for f in self._carte(i)["fiches"] if f["prompt"] == prompt), None)
        if f is None:
            raise KeyError(f"prompt absent de la carte {i} : {prompt[:60]}")
        bif, val = f.get("bifurcation") or {}, f.get("valeur_ajoutee") or {}
        ci = val.get("ecart_ci") or [None, None]
        verdict = val.get("verdict")
        testee = verdict is not None and not verdict.startswith("NON TESTABLE")
        return {
            "multimodal": 1.0 if f.get("multimodal") else 0.0,
            "dip": _nan(f.get("dip")),
            "p": _nan(f.get("p")),
            "suites_distinctes": _nan(f.get("suites_distinctes")),
            "entropie_moyenne": _nan(f.get("entropie_moyenne")),
            "prefixe_k": _nan(bif.get("k")),
            "prefixe_gain_max": _nan(bif.get("gain_max")),
            "valeur_ecart": _nan(val.get("ecart")),
            "valeur_ecart_bas": _nan(ci[0]),
            "valeur_ajoutee": (1.0 if verdict == "VALEUR AJOUTÉE" else 0.0) if testee else float("nan"),
        }

    def matrice_signatures(self, i: int) -> Tuple[List[str], np.ndarray, Tuple[str, ...]]:
        """(prompts, matrice (P, len(CHAMPS_SIGNATURE)), noms des colonnes) de la carte i."""
        prompts = [f["prompt"] for f in self._carte(i)["fiches"]]
        X = np.array([[self.signature(i, p)[c] for c in CHAMPS_SIGNATURE] for p in prompts], float)
        return prompts, X, CHAMPS_SIGNATURE

    def resume_carte(self, i: int) -> Dict[str, Any]:
        c, par = self._carte(i), self.parametres_carte(i)
        fi = c["fiches"]
        multi = [f for f in fi if f.get("multimodal")]
        verdicts = Counter((f.get("valeur_ajoutee") or {}).get("verdict", "non calculée") for f in multi)
        gains = [f["bifurcation"]["gain_max"] for f in multi if f.get("bifurcation")]
        k_trouves = [f for f in multi if (f.get("bifurcation") or {}).get("k") is not None]
        return {
            "nom": self.cartes[i]["nom"],
            "modele": par.get("modele"),
            "couche": par.get("couche"),
            "representation": par.get("representation"),
            "n_prompts": len(fi),
            "regimes": c["regimes"],
            "n_multimodaux": len(multi),
            "valeur_ajoutee": dict(verdicts),
            "prefixe_k_trouve": len(k_trouves),
            "prefixe_gain_max_median": float(np.median(gains)) if gains else None,
            "suites_distinctes_median": float(np.median([f["suites_distinctes"] for f in fi])) if fi else None,
        }

    def accord_cartes(self, i: int, j: int) -> Dict[str, Any]:
        """Les mêmes prompts sont-ils multimodaux dans les deux cartes (deux modèles, par exemple) ? Sur les prompts
        communs : prompts multimodaux partagés, nombre attendu par hasard et p hypergéométrique (au moins autant de
        partagés), indice de Jaccard. La comparabilité ignore le modèle et la couche, qui diffèrent entre modèles."""
        fa = {f["prompt"]: bool(f.get("multimodal")) for f in self._carte(i)["fiches"]}
        fb = {f["prompt"]: bool(f.get("multimodal")) for f in self._carte(j)["fiches"]}
        communs = [p for p in fa if p in fb]
        ma = {p for p in communs if fa[p]}
        mb = {p for p in communs if fb[p]}
        k, n = len(ma & mb), len(communs)
        union = ma | mb
        return {
            "n_communs": n,
            "multi_a": len(ma),
            "multi_b": len(mb),
            "partages": k,
            "attendu": len(ma) * len(mb) / n if n else None,
            "p_hypergeometrique": p_hypergeometrique(n, len(ma), len(mb), k) if n else None,
            "jaccard": k / len(union) if union else None,
            "prompts_partages": [p for p in communs if p in ma and p in mb],
            "comparabilite": comparer_provenances(self.parametres_carte(i), self.parametres_carte(j), CLES_CARTES,
                                                  ignorer=("modele", "couche")),
        }

    def exporter_carte(self, i: int, fichier: Union[str, Path]) -> None:
        """Une carte et sa provenance dans un fichier JSON (pour fusionner des cartes faites dans des processus séparés)."""
        Path(fichier).write_text(json.dumps(self.cartes[i], indent=1, ensure_ascii=False))

    def importer_carte(self, fichier: Union[str, Path]) -> int:
        rec = json.loads(Path(fichier).read_text())
        i = self.add_carte(rec["carte"], rec.get("provenance"), rec.get("nom"))
        self.cartes[i]["cree"] = rec.get("cree", self.cartes[i]["cree"])     # date de construction, pas d'import
        return i

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
        Mesures historiques (v0.1) : pour l'espace caché, voir signature().
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

        # Cartes (v0.4) : JSON seul, rien n'est exécuté à la lecture
        with open(path / "atlas_cartes.json", "w", encoding="utf-8") as f:
            json.dump(self.cartes, f, indent=1, ensure_ascii=False)

        # Données complètes (pickle : ne charger que des atlas de confiance)
        with open(path / "atlas_data.pkl", "wb") as f:
            pickle.dump(self.entries, f)

        print(f"[Atlas] Sauvegardé → {path}")

    @classmethod
    def load(cls, directory: str) -> "BifurcationAtlas":
        """Charge un atlas v0.2, v0.3 ou v0.4. Le pickle des entrées historiques
        exécute du code : ne charger que des atlas de confiance (les cartes sont
        en JSON)."""
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
        # Atlas v0.2 et v0.3 : pas de cartes
        fc = path / "atlas_cartes.json"
        atlas.cartes = json.loads(fc.read_text(encoding="utf-8")) if fc.exists() else []
        atlas.metadata["n_cartes"] = len(atlas.cartes)
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
