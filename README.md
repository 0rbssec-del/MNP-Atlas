# MNP-Atlas

Tools and pre-registered experiments for studying how sampled trajectories of language models diverge, with a
replication of Akarlar (2026), *Hallucination as Trajectory Commitment* (arXiv 2604.15400).

Author: Erick P., independent researcher. Most internal documents are written in French; this README and the
replication note are in English.

## Main result: replication of Akarlar (2026)

The paper reports that an activation patch turns a correct trajectory into a hallucination in 87.5% of trials but
corrects a hallucinated one in only 33.3%, and reads this asymmetry as a hallucination attractor. Using the authors'
code, model (Qwen2.5-1.5B), prompts and classifier:

| Finding | Evidence | Status |
|---|---|---|
| In the authors' own raw data, the outcome distribution does not depend on the source run | 107/452/113 vs 104/457/111 correct/hallucinated/other, χ² p = 0.96 | established |
| The published patch sets the patched step's distribution exactly to the source's | Jensen–Shannon divergence 0 in 1,232 of 1,232 cases | established |
| Flip rates match the no-patch rate of each label | 0.64 vs 0.70 hallucinated; 0.12 vs 0.15 correct (no detectable difference, about ±10 points) | established |
| No layer effect in the paper's Table 4 | χ² p = 0.71; peaks compatible with the maximum of 28 noisy cells | established |
| Direct test, persistent patch, three models (Qwen2.5-1.5B, Qwen2.5-7B, OLMo-2-7B) | no asymmetry averaged over layers; wide intervals once prompts are resampled | exploratory |
| Confirmatory test with a same-class control (Qwen2.5-7B, layer 24 fixed in advance, 46 prompts) | A* = −0.37 [−0.75, −0.04] in log-odds: correction easier than corruption, the reverse of the paper's claim; indeterminate with the strict classifier variant | fragile |

The published asymmetry is explained by classifier base rates. The direct tests find no sign of the claimed basin;
the reverse asymmetry in the confirmatory test is not robust to the classifier variant. The paper's lexical
classifier has not been validated against hand labels, which is the main open limitation.

The full replication note (PDF) is available from the author on request and will be posted on arXiv.

## Pre-registration

Each experiment's decision criteria were committed **before** it was run. The commit is the timestamp.

| Experiment | Criteria (commit) | Results (commit) | Files |
|---|---|---|---|
| Replication of the published experiment (P1, P1b, P2) | `5dd1a6e` (amendments `bdec38e`) | `df3abfb`, `bdec38e` | `experiments/replication_akarlar.py`, `experiments/REPLICATION_AKARLAR.md` |
| Test on the paper's Table 4 | — (computed on published data) | `7a2c64e` | `experiments/akarlar_tableau4.py` |
| Direct basin test, 8 prompts | `0af62ba` | `8142dff` | `experiments/bassin_akarlar.py` |
| Confirmation, 27 bifurcating prompts | `053d9c2` | `9bd8521` | `experiments/BASSIN_27.md` |
| Two larger models (Qwen2.5-7B, OLMo-2-7B) | `ad24a0c` | `da0727b` | `experiments/BASSIN_7B.md` |
| Confirmatory test with same-class control | `adb5dd4` | `7dcbb75` | `experiments/BASSIN_CONFIRMATION.md`, `experiments/bassin_confirmation.py` |

Raw outputs (every sampled continuation and its label) are in `experiments/resultats/`. The confirmatory test was run
twice on Kaggle with the same seed; both runs produced identical continuations and labels
(`kaggle_2026-11-03/log_bassin_confirmation*.txt`).

## Reproducing

Requirements: Python ≥ 3.10 and the packages in `requirements.txt`. The 1.5B experiments run on one 16 GB GPU (or,
slowly, on CPU); the 7B experiments need two 16 GB GPUs (for example Kaggle's free T4 × 2) and use `--repartir`
to split the model across them.

```bash
pip install -r requirements.txt
pytest -q tests/test_bassin_akarlar.py            # set-up checks on a tiny random model (CPU)

# Direct basin test on the 27 bifurcating prompts (Qwen2.5-1.5B, ~25 min on a T4)
python experiments/bassin_akarlar.py --prompts experiments/data/akarlar_bifurcating_prompts.json \
    --seed 20261103 --out bassin_27.json

# Two larger models (~40 min each on two T4)
python experiments/bassin_akarlar.py --model Qwen/Qwen2.5-7B --prompts experiments/data/akarlar_bifurcating_prompts.json \
    --seed 20261104 --repartir --couches-relatives --lot 96 --min-paires 40 --out bassin_qwen7b.json

# Confirmatory test with same-class control (~65 min on two T4)
python experiments/bassin_confirmation.py --model Qwen/Qwen2.5-7B --repartir --lot 96 --out bassin_confirmation.json
```

Each script prints its controls and its verdict under the pre-registered rule, and writes every continuation to the
output JSON.

## Repository layout

| Path | Content |
|---|---|
| `trajectory_bifurcation_core.py` | Engine: sampling with common random numbers (Gumbel-max, one seed per row and step), hidden-state capture, activation patching (`docs/ENGINE.md`) |
| `atlas/` | Atlas layer built on the engine: hidden-space maps and a results index (`docs/ATLAS.md`) |
| `experiments/` | Replication scripts, pre-registration documents, and other experiments (dose–response, multi-prompt) |
| `experiments/data/` | Akarlar's prompts and classifier word lists, extracted without executing his code from github.com/akarlaraytu/trajectory-commitment (MIT licence) |
| `experiments/resultats/` | Raw results and Kaggle logs |
| `tests/` | Unit and set-up tests (CPU, tiny random models) |

## Limitations

- The paper's lexical classifier is kept as is; its labels have not been validated by hand.
- The direct tests use one fixed layer for the confirmatory verdict and few models; prompts are those of the paper.
- The 7B models run in float16 split across two T4 GPUs.

## Use of AI assistance

Code, analyses and writing were produced with the help of an AI assistant (Claude, Anthropic). All decision criteria
were fixed in commits before the corresponding runs, and every number reported can be recomputed from the raw files.

## Contact

Open an issue on this repository.
