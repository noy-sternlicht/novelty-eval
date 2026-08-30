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
  - [Evaluating a judge](#evaluating-a-judge)
  - [Reproducing paper results](#reproducing-paper-results)
    - [Building the benchmark](#building-the-benchmark)
    - [Retrieval](#retrieval)
    - [Running the judges](#running-the-judges)
    - [Ablations](#ablations)
    - [Figures and tables](#figures-and-tables)
  - [Repository structure](#repository-structure)
  - [Citation](#citation)
  - [Authors](#authors)
  - [License](#license)
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

The external novelty judges from prior work ([AI-Scientist](https://arxiv.org/abs/2408.06292), [Idea-Novelty-Checker](https://arxiv.org/abs/2506.22026)) need a conflicting `transformers` pin and live in a separate environment:

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
This section describes how to use our automatic data collection pipeline to recreate the benchmark using more recent publications.

TODO

## Evaluating a judge

To run a single judge configuration over one benchmark file:

```bash
python3 src/novelty_eval/run_benchmark.py \
  --config src/novelty_eval/config/accuracy_test.yaml \  # Judge configuration
  --set test_inputs=data/human-only/pairwise.yaml \      # Benchmark file to evaluate on
  --set llm_engine=gpt-5.4 \                             # Judge backbone
  --set test_mode=pairwise                               # "pairwise" or "pointwise"
```

Anything in the config can be overridden with a repeatable `--set KEY=VALUE`; see [`config/accuracy_test.yaml`](src/novelty_eval/config/accuracy_test.yaml) for the full set of knobs (reasoning effort, number of samples per decision, retrieval cache, concurrency).

Each run writes a timestamped directory under `$OUTPUT_DIR` containing:

* `scores.json` — the judge's verdict on every instance.
* `accuracy_report.txt` — the metrics (pairwise soft/strict accuracy, pointwise per-class and macro F1).
* a Markdown debug report with the prompt, reasoning and verdict behind each decision.
* a cost report (tokens, cache hits, estimated USD).

To run many configurations at once, use the sweep runner:

```bash
python3 src/novelty_eval/run_benchmark_sweep.py \
  --config src/novelty_eval/config/accuracy_sweep_config.yaml \
  --parallel
```

**Evaluating your own judge.** Judges that are not prompted LLMs plug in behind a narrow seam: subclass `BaselineRunner` ([`baselines/base.py`](src/novelty_eval/baselines/base.py)), return a prediction, and register the class in [`baselines/registry.py`](src/novelty_eval/baselines/registry.py). Everything above the seam — the run loop, metrics, reports — is unchanged. [`baselines/ai_scientist`](src/novelty_eval/baselines/ai_scientist) and [`baselines/scideator`](src/novelty_eval/baselines/scideator) are worked examples.

## Reproducing paper results

You can enter this pipeline at any stage: the released `data/` lets you skip benchmark construction, and the released run artifacts (TODO: link) let you skip the judge runs.

### Building the benchmark

1. **Mine novelty signals from ICLR reviews.** Downloads submissions and reviews via the OpenReview API, prompts an LLM to extract per-reviewer novelty signals, and filters by rating percentile within each ICLR primary area:
   ```bash
   python3 src/novelty_eval/benchmark_data/fetch_iclr_data.py \
     --year 2026 \             # ICLR year to process
     --percentile 10.0 \       # Top/bottom percentile within each primary area
     --model claude-opus-4-6 \ # Model used for novelty-signal extraction
     --output_dir iclr_data
   ```
   Re-running this on a newer venue produces a fresh benchmark outside the judges' training cutoff.

2. **Assemble evaluation instances.** Pairs the mined ideas into benchmark instances, optionally generating the low-novelty side with an LLM ideator:
   ```bash
   python3 src/novelty_eval/benchmark_data/create_benchmark_instances.py \
     --config src/novelty_eval/benchmark_data/config/TODO.yaml
   ```
   The configs under [`benchmark_data/config/`](src/novelty_eval/benchmark_data/config) correspond to the released data files: TODO (map config → data file).

   [`benchmark_data/paper_blocklist.yaml`](src/novelty_eval/benchmark_data/paper_blocklist.yaml) lists papers excluded for tripping LLM safety filters. The released data is already filtered; `analysis/filtering.py` re-applies the blocklist when metrics are computed on freshly built instances.

### Retrieval

The retrieval conditions give the judge a "Related Work" context. Candidates are retrieved once into a disk-backed cache, then read from it at judging time — no online search happens during a benchmark run:

```bash
python3 src/novelty_eval/retrieval/retrieve_candidates.py \
  --config src/novelty_eval/retrieval/retrieve_candidates.yaml
```

Pass the resulting cache to a run with `--set retrieval_cache_file=<path>`. Pairwise caches are converted to pointwise ones with `retrieval/flatten_retrieval_cache.py`.

### Running the judges

TODO: the sweep config used for the main results table, and how to reproduce it.

```bash
./scripts/run_accuracy_sweep.sh
```

### Ablations

All ablation conditions — prompt wording, reasoning effort, retrieval, idea format, negatives source — are defined in [`ablation/ablations.yaml`](src/novelty_eval/ablation/ablations.yaml), the single source of truth for what each experiment varies. To run them:

```bash
./scripts/run_ablations.sh \
  --ablations current,TODO \  # Which ablations to run (default: all)
  --model gpt-5.4 \           # Judge backbone
  --n-runs 3                  # Samples per decision
```

Merge the resulting run directories into one comparison report and the chart data the figure scripts consume:

```bash
python3 src/novelty_eval/ablation/merge_ablation_runs.py \
  --config src/novelty_eval/ablation/configs/merge/TODO.yaml
```

### Figures and tables

The figure scripts read the merged chart data, so they redraw without re-running a merge or a bootstrap:

```bash
python3 src/novelty_eval/figures/chart_two_track.py \
  --track "Human-Only"=<merged_dir> --track "Human+Generated"=<merged_dir> \
  --config src/novelty_eval/figures/configs/TODO.yaml \
  --out-dir output/paper_figures
```

TODO: table mapping each paper figure and table to its script and config.

## Repository structure

```
data/                     benchmark instances
src/novelty_eval/
  benchmark_data/         ICLR mining and instance construction
  retrieval/              related-work retrieval and caching
  run_benchmark.py        evaluation entry point
  judge.py, metrics.py    judge invocation and scoring
  ablation/               ablation definitions, orchestration, merging
  analysis/               artifact reading, filtering, significance testing
  figures/                camera-ready figures and tables
  baselines/              novelty judges from prior work
scripts/                  shell entry points and reporting utilities
tests/                    pytest suite; runs without API keys
```

[`src/novelty_eval/README.md`](src/novelty_eval/README.md) documents every module.

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
TODO
```

## Authors

* TODO

## License

TODO
