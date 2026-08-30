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

The same pipeline that produced these files can be re-run on a newer conference cycle. See instructions here:

[**Rebuilding the benchmark**](src/novelty_eval/benchmark_data/README.md)

## Judge Evaluation

A judge run takes one benchmark file from [data/](data/) and one backbone model, and reports how well that judge separates `D_h` from `D_l`. The configs in [config/](src/novelty_eval/config/) pin the instance paths of our own runs, so point them at a released file with `--set`:

```bash
python src/novelty_eval/run_benchmark.py \
    --config src/novelty_eval/config/accuracy_test.yaml \
    --set test_inputs=data/human-only/pairwise.yaml \
    --set test_mode=pairwise \
    --set llm_engine=claude-opus-4-6 \
    --set reasoning_effort=high \
    --set output_dir=output \
    --set num_instances=5
```

`num_instances=5` caps the run at five instances — enough to check the setup end to end for a few cents. Drop it to evaluate the whole file. The pointwise format is the same command with the pointwise file and mode:

```bash
    --set test_inputs=data/human-only/pointwise.yaml \
    --set test_mode=pointwise
```



## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
TODO
```

## Authors

* TODO
