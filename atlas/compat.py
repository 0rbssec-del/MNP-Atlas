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
