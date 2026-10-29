# trajectory_bifurcation_core.py
# Core v0.1.5 – Trajectory Bifurcation Engine
# Fonctionnalités :
#   - Génération d'ensembles de trajectoires à prompt fixe (KV-cache, graines)
#   - Divergences : Jensen-Shannon (top-k + masse résiduelle) + distances
#     résiduelles (L2 et cosinus)
#   - Détection de bifurcation (seuil relatif + plancher absolu)
#   - Test de réversibilité par activation patching (hooks), échantillonnage
#     couplé et métrique déterministe au niveau des logits
#
# Changements v0.1.3 → v0.1.4 (voir le message de commit pour le détail) :
#   - Le patch persiste dans le KV-cache (avant : actif sur la 1re passe
#     seulement, donc effet limité au 1er token).
#   - Continuations comparées sur les nouveaux tokens seulement, avec la même
#     graine pour la version propre et la version patchée.
#   - États capturés par hooks sur la sortie brute des couches (avant :
#     hidden_states[-1] inclut la norme finale → double norme au patch).
#   - Trajectory.text ne contient plus que la complétion.
#
# Changements v0.1.4 → v0.1.5 :
#   - Le patching n'a plus lieu à t* même (où les paires ne diffèrent souvent
#     qu'au dernier token, recovery trivial) mais au premier t >= t* offrant
#     assez de paires informatives (choose_patch_time).
#   - recovery multi-pas : la continuation de la source est forcée sur
#     `forced_horizon` tokens, JS exacte par pas et log-vraisemblance récupérée.

from __future__ import annotations
import contextlib
import torch
import torch.nn.functional as F
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Iterator
import numpy as np
from tqdm import tqdm

TopK = Tuple[np.ndarray, np.ndarray, float]

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
    # Sortie brute de la couche l (avant norme finale), dernière position
    hidden_states: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    # (top_indices, top_probs, remaining_mass)
    topk_distributions: List[TopK] = field(default_factory=list)

    @property
    def generated_ids(self) -> List[int]:
        return self.token_ids[self.prompt_len:]

@dataclass
class Continuation:
    token_ids: List[int]
    text: str
    first_distribution: TopK

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
# Moteur principal
# --------------------------------------------------------------------------

class TrajectoryBifurcationEngine:
    # Planchers absolus par mesure : sans eux, une ligne de base ~0 (premiers
    # tokens identiques, ex. "<think>" chez Qwen3) donne un seuil ~0 et une
    # bifurcation « détectée » dès t = min_steps. Valeurs heuristiques.
    DEFAULT_ABS_FLOOR = {"js": 0.05, "residual": 0.05, "residual_l2": 0.0}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        layers_to_track: Optional[List[int]] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        topk_for_divergence: int = 256,
        seed: int = 0,
        trust_remote_code: bool = False,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.topk_for_divergence = topk_for_divergence
        self.seed = seed

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if torch_dtype is None:
            # bf16 sur CPU est lent sans AVX512-BF16/AMX : float32 par défaut
            torch_dtype = (
                torch.bfloat16 if self.device.type == "cuda" else torch.float32
            )

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

        print(f"[Core] Couches suivies : {self.layers_to_track}")
        print(f"[Core] Top-k pour divergences : {self.topk_for_divergence}")

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
        """Capture la sortie de chaque couche à la dernière position.

        On n'utilise pas output_hidden_states : chez HF, hidden_states[-1]
        inclut la norme finale, ce qui ne correspond pas au point d'injection.
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
    # Génération
    # ------------------------------------------------------------------

    def _make_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return g

    def _sample(
        self, logits: torch.Tensor, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
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

    @torch.no_grad()
    def _decode_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        seed: Optional[int],
        record_layers: Optional[List[int]] = None,
        patch: Optional[Tuple[int, int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Boucle d'échantillonnage commune (ensemble et continuations).

        Entropie et top-k sont mesurés sur la distribution du modèle (T = 1) ;
        l'échantillonnage utilise temperature / top_p.
        """
        record_layers = record_layers or []
        generator = self._make_generator(seed)
        new_tokens: List[int] = []
        entropies: List[float] = []
        topk_dists: List[TopK] = []
        hiddens: Dict[int, List[np.ndarray]] = {l: [] for l in record_layers}

        with contextlib.ExitStack() as stack:
            captured = stack.enter_context(self._capture_hooks(record_layers))
            if patch is not None:
                stack.enter_context(self._patch_hook(*patch))

            past = None
            cur = input_ids
            for _ in range(max_new_tokens):
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
                topk_dists.append(self._topk(probs))
                for l in record_layers:
                    hiddens[l].append(captured[l].numpy())

                next_token = self._sample(logits, generator)
                token = int(next_token.item())
                new_tokens.append(token)
                if token in self.eos_token_ids:
                    break
                cur = next_token.view(1, 1)

        return {
            "new_tokens": new_tokens,
            "entropies": entropies,
            "topk": topk_dists,
            "hidden_states": hiddens,
        }

    def build_prompt_ids(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
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
        )

    def generate_ensemble(
        self,
        prompt: str,
        n_trajectories: int = 16,
        system_prompt: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> List[Trajectory]:
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
        first_diff = cls.first_diff_step(src, tgt, t_patch)
        if first_diff is None:
            return "negative_control"
        if first_diff == t_patch - 1:
            return "positive_control"
        return "informative"

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

            js_over_time.append(
                float(
                    np.mean(
                        [
                            self.js_divergence_topk(
                                trajectories[i].topk_distributions[t],
                                trajectories[j].topk_distributions[t],
                            )
                            for i, j in pairs
                        ]
                    )
                )
            )

        ent_matrix = np.array(
            [t.entropy_shannon[:min_len] for t in trajectories]
        )

        return {
            "min_len": min_len,
            "residual_dist_over_time": {
                l: np.array(v) for l, v in residual_dists.items()
            },
            "residual_cos_over_time": {
                l: np.array(v) for l, v in residual_cos.items()
            },
            "js_over_time": np.array(js_over_time),
            "frac_pairs_same_prefix": np.array(frac_same_prefix),
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
        """measure : "js", "residual" (cosinus) ou "residual_l2"."""
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
    ) -> Continuation:
        """Continue `prefix_ids`, éventuellement avec patch (couche, pos, vecteur)."""
        out = self._decode_loop(
            prefix_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            seed=seed,
            patch=patch,
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
            (layer, position, torch.from_numpy(np.asarray(vector))),
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
                1.0 - float(js_ps.sum()) / den_js if den_js > 1e-4 else None
            ),
            "recovery_loglik": (
                (ll_patch - ll_clean) / den_ll if abs(den_ll) > 1e-3 else None
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
    ) -> Optional[int]:
        """Premier t >= t_min offrant au moins n_pairs paires informatives.

        À t* même, les paires ne diffèrent souvent qu'au dernier token
        (témoins positifs) : patcher là ne teste rien.
        """
        n = len(trajectories)
        lengths = [len(t.entropy_shannon) for t in trajectories]
        for t in range(max(1, t_min), max(lengths)):
            count = sum(
                1
                for i in range(n)
                for j in range(i + 1, n)
                if min(lengths[i], lengths[j]) > t
                and self.classify_pair(trajectories[i], trajectories[j], t)
                == "informative"
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
        source_hidden = torch.from_numpy(src.hidden_states[layer][t_patch])

        if seed is None:
            seed = self.seed + 7919 * (source_idx + 1) + target_idx
        # Même graine : toute différence entre les deux continuations vient du
        # patch, pas du bruit d'échantillonnage
        clean = self.continue_from_prefix(
            prefix_ids, max_continuation_tokens, seed=seed
        )
        patched = self.continue_from_prefix(
            prefix_ids,
            max_continuation_tokens,
            seed=seed,
            patch=(layer, position, source_hidden),
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
    ) -> List[PatchResult]:
        """Patch source → cible sur des paires tirées au hasard.

        Paires informatives : préfixes différents avant la position patchée.
        Témoins négatifs : préfixes identiques (patch nul ; continuations
        identiques attendues). Témoins positifs : seul le dernier token du
        préfixe diffère (recovery = 1 attendu). Jusqu'à n_controls de chaque.
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

        quota = {
            "informative": n_pairs,
            "positive_control": n_controls,
            "negative_control": n_controls,
        }
        selected: Dict[str, List[Tuple[int, int]]] = {k: [] for k in quota}
        seen = set()
        for i, j in ordered:
            if frozenset((i, j)) in seen:
                continue
            seen.add(frozenset((i, j)))
            if min(len(trajectories[i].hidden_states[layer]),
                   len(trajectories[j].hidden_states[layer])) <= t_patch:
                continue
            kind = self.classify_pair(trajectories[i], trajectories[j], t_patch)
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

        info = [r for r in patch_results if r.pair_kind == "informative"]
        pos = [r for r in patch_results if r.pair_kind == "positive_control"]
        neg = [r for r in patch_results if r.pair_kind == "negative_control"]
        return {
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
        kind = r.pair_kind
        fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"
        print(
            f"[{kind}] {r.source_traj_idx}→{r.target_traj_idx} "
            f"recovery={fmt(r.recovery)} "
            f"multi-pas={fmt(r.recovery_multistep)} "
            f"loglik={fmt(r.recovery_loglik)} "
            f"overlap={r.final_token_overlap:.2f}"
        )
