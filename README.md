# TODO: Paper Title

[![Arxiv](https://img.shields.io/badge/Arxiv-TODO-red?style=flat-square&logo=arxiv&logoColor=white)](TODO)
[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Project Page](https://img.shields.io/badge/Project%20Page-Here-green?style=flat-square&logo=github)](TODO)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-Here-yellow?style=flat-square)](https://huggingface.co/datasets/noystl/novelty-judge-bench)

TODO: short abstract-style paragraph.

<p align="center">
  <img src="TODO-overview.png" alt="Overview" />
</p>

## Table of Contents

<!-- TOC -->
- [TODO: Paper Title](#todo-paper-title)
  - [Table of Contents](#table-of-contents)
  - [Getting started](#getting-started)
    - [Setting up API keys](#setting-up-api-keys)
  - [Novelty Evaluation Data](#novelty-evaluation-data)
    - [🤗 Hugging Face](#-hugging-face)
    - [Automatic Data Collection](#automatic-data-collection)
  - [Judge Evaluation](#judge-evaluation)
  - [Reproducing the Paper's Experiments](#reproducing-the-papers-experiments)
  - [Citation](#citation)
  - [Authors](#authors)
<!-- TOC -->

## Getting started

1. Clone this repository:
   ```bash
   git clone TODO-REPO-URL
   ```
2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv myenv
   source ./myenv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Point the code at the repository:
   ```bash
   export PYTHONPATH="$PWD/src"
   export SECRETS="$PWD/secrets.toml"
   export OUTPUT_DIR="$PWD/output"
   ```

The external novelty judges from prior work ([AI-Scientist](https://arxiv.org/abs/2408.06292), [Idea-Novelty-Checker](https://arxiv.org/abs/2506.22026)) follow the implementations released at [simra-shahid/idea_novelty_checker](https://github.com/simra-shahid/idea_novelty_checker). They need a conflicting `transformers` pin and live in a separate environment:

```bash
python3.12 -m venv myenv-baselines
./myenv-baselines/bin/pip install -r requirements-baselines.txt
```

### Setting up API keys

The judges and the benchmark-construction pipeline call external APIs. Keys go in a `secrets.toml` file at the project root (the location `$SECRETS` points at):

```toml
openai_key = "your-openai-api-key"
anthropic_key = "your-anthropic-api-key"
semantic_scholar_key = "your-s2-api-key"   # retrieval only
```

Rebuilding the benchmark from scratch additionally needs OpenReview credentials:

```bash
export OPENREVIEW_USERNAME="your-openreview-email"
export OPENREVIEW_PASSWORD="your-openreview-password"
```

## Novelty Evaluation Data

Our benchmark consists of two data setups: **Human-Only** and **Human+Generated**. Both setups consist of two separate idea "pools":  **high-novelty pool `D_h`** and **low-novelty pool `D_l`** where ideas from `D_h` can be assumed to be more novel than ones in `D_l`.

| Setup | `D_h` (novel) | `D_l` (lower-novelty) |
|---|---|---|
| **Human-Only** (`data/human-only/`) | 154 **validated-novel** | 146 **validated-lower-novelty** |
| **Human+Generated** (`data/human-plus-generated/`) | 154 **validated-novel** | 154 **weak-labeled-lower-novelty** |

Where: 
* **validated-novel** (both setups) - ICLR 2026 submissions whose reviewers explicitly praised the originality of the contribution itself and none disputed it, accepted, in the top rating decile of their primary area, average contribution score >= 3.0.
* **validated-lower-novelty** (Human-Only) - the mirror image: reviewers faulted the originality and none praised it, rejected, in the bottom rating decile of their primary area, average contribution score <= 2.0.
* **weak-labeled-lower-novelty** (Human+Generated) - ideas from Claude Sonnet 4.5, prompted simply to propose a novel idea with no scaffold, literature access, or tools. These carry no review-based label; they are *assumed* to fall below the human high-novelty bar.

Each setup is instantiated in two evaluation formats:

* **pointwise** — one idea per row, binary label: is this idea novel?
* **pairwise** — one `D_h` idea against one `D_l` idea; which of the two is more novel?

and in two idea formats:

* **abstract** (default) — the idea as a free-form abstract.
* **plan** (`*_plan` configs) — the same ideas normalized into a two-field *purpose* / *mechanism* research plan.

`data/human-plus-generated/backbone-*/` regenerates `D_l` with other ideation backbones (`gpt-5.1`, `gpt-5.4`, `opus-4-5`) and is used in the negatives-source experiment.

### 🤗 Hugging Face

The same instances are released on the Hub as [noystl/novelty-judge-bench](https://huggingface.co/datasets/noystl/novelty-judge-bench)

```python
from datasets import load_dataset

ds = load_dataset("noystl/novelty-judge-bench", "human-plus-generated_pointwise", split="test")
```

### Automatic Data Collection

The same pipeline that produced these files can be re-run on a newer conference cycle. See instructions here:

[**Rebuilding the benchmark**](src/novelty_eval/benchmark_data/README.md)

## Judge Evaluation

[example_eval.yaml](src/novelty_eval/config/example_eval.yaml) runs a single judge over a single released benchmark file, under the baseline configuration of the paper: a verdict plus reasoning, three samples per decision, high reasoning effort, no tools and no related work. 

```bash
python src/novelty_eval/run_benchmark.py \
    --config src/novelty_eval/config/example_eval.yaml
```

Comparing several judges, or the same judge under a controlled change, is a *sweep*: [example_sweep.yaml](src/novelty_eval/config/example_sweep.yaml) sets a shared `base_config` and one entry per run. Each entry becomes its own judge run, and the results are collected into a cross-config `sweep_comparison_report.md`.

```bash
python src/novelty_eval/run_benchmark_sweep.py \
    --config src/novelty_eval/config/example_sweep.yaml \
    --parallel --skip-on-error
```

## Reproducing the Paper's Experiments

Each experiment below is one ablation run plus a figure script. [ablations.yaml](src/novelty_eval/ablation/ablations.yaml) is the registry of conditions; [figures/configs/](src/novelty_eval/figures/configs/) holds the paper's labels and panel ordering.

| Experiment | Run | Figure |
|---|---|---|
| **Controlled changes** — novelty criterion, prior-work access, reasoning effort, idea format | `./scripts/eval/run_ablations.sh --ablations current,vague_criterion,no_criteria_prompt,low_judge_reasoning,retrieval` | [figures/two_track_figure.ipynb](src/novelty_eval/figures/two_track_figure.ipynb) |
| **Prompt design case study** | `./scripts/eval/run_ablations.sh --ablations ai_researcher_base,ai_researcher_no_review,ai_researcher_comparative` | [figures/prompt_design_figure.ipynb](src/novelty_eval/figures/prompt_design_figure.ipynb) |
| **Retrieval face-off** — a strong web-search judge against the retrieval-aided ones | `./scripts/eval/run_retrieval_faceoff.sh sample`, then `submit`, then `poll` | — |
| **Negatives source** — `D_l` regenerated by four ideation backbones | ablations over `data/human-plus-generated/backbone-*/` | `chart_negatives_source.py --config figures/configs/negatives_source_figure.yaml` |
| **Dedicated novelty judges** | `run_benchmark.py --config src/novelty_eval/config/accuracy_test_{ai_scientist,scideator}.yaml` (in `myenv-baselines`) | — |
| **Cost–benefit** | [cost_efficiency.ipynb](src/novelty_eval/figures/cost_efficiency.ipynb) | — |

Ablation runs write per-condition artifacts; the figure scripts read a *merge* over them:

```bash
python src/novelty_eval/ablation/merge_ablation_runs.py \
    --config src/novelty_eval/ablation/configs/merge/merge_config_vanilla.yaml
```


 **Note: Retrieval conditions need a precompute pass.** `retrieval/retrieve_candidates.py` builds the candidate cache, and `flatten_retrieval_cache.py` derives the pointwise cache from the pairwise one. Retrieval happens once per track, so its cost is a flat term rather than a per-decision one.

A file-by-file map of the codebase lives in [src/novelty_eval/README.md](src/novelty_eval/README.md).

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
TODO
```

## Authors

* TODO
