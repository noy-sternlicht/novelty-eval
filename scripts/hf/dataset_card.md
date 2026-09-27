---
pretty_name: Novelty Judge Bench
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
tags:
- novelty
- peer-review
- llm-as-a-judge
- ai-generated-ideas
- research-ideation
- benchmark
size_categories:
- n<1K
annotations_creators:
- expert-generated
- machine-generated
source_datasets:
- original
configs:
{{CONFIGS_YAML}}
---

# TODO-TITLE

Benchmark instances for **TODO-TITLE** (paper link: TODO).

TODO-DESC

Code: TODO-REPO-URL

## Setups

Both setups share their **high-novelty pool `D_+`** (154 top-decile accepted ICLR 2026
submissions for which a majority of reviewers explicitly praised the novelty of the
contribution and none disputed it) and differ in where the **low-novelty pool `D_-`** comes
from:

| Setup | `D_+` | `D_-` |
|---|---|---|
| `human-only` | 154 ICLR submissions | 145 ICLR submissions whose reviewers faulted the originality |
| `human-plus-generated` | the same 154 | 154 ideas from a plain LLM ideator (no scaffold, no tools, no literature access) |


## Formats

Each setup is instantiated in two evaluation formats:

* **pointwise** — one idea per row, binary label: is this idea novel?
* **pairwise** — one `D_+` idea against one `D_-` idea; which of the two is more novel?

and in two idea formats:

* **abstract** (default) — the idea as a free-form abstract.
* **plan** (`*_plan` configs) — the same ideas normalized into a two-field
  *purpose* / *mechanism* research plan.

The `*_backbone-*` configs regenerate `D_-` with a different ideation backbone and are used
in the negatives-source experiment.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("noystl/novelty-judge-bench", "human-plus-generated_pointwise", split="test")
print(ds[0]["idea"], ds[0]["label"])
```


## Fields

### pointwise configs

| Field | Type | Description                                                      |
|---|---|------------------------------------------------------------------|
| `id` | int32 | Instance id, unique within the config                            |
| `iclr_area` | string | The idea's corresponding ICLR primary area                       |
| `idea` | string | The idea to judge (abstract or plan form, per config)            |
| `label` | string | `POSITIVE` (novel) or `NEGATIVE` (not novel) — the ground truth  |
| `idea_source` | string | `human` (an ICLR submission) or `generated` (LLM ideator output) |
| `title` | string | Submission title, or a synthetic identifier for generated ideas  |
| `rating` | float64 | Mean reviewer rating; `null` for generated ideas                 |
| `contribution` | float64 | Mean reviewer contribution score; `null` for generated ideas     |
| `positive_signals` | list[string] | Reviewer excerpts praising originality                           |
| `negative_signals` | list[string] | Reviewer excerpts faulting originality                           |

### pairwise configs

| Field | Type | Description |
|---|---|---|
| `id` | int32 | Instance id, unique within the config |
| `iclr_area` | string | The ICLR primary area shared by both ideas in the pair |
| `idea_a`, `idea_b` | string | The two ideas to compare |
| `expected_winner` | int8 | `0` if `idea_a` is the more novel one, `1` if `idea_b` |
| `label_a`, `label_b` | string | `POSITIVE` / `NEGATIVE` for each side |
| `idea_source_a`, `idea_source_b` | string | `human` or `generated` |
| `title_a`, `title_b` | string | Per-side titles |
| `rating_a`, `rating_b` | float64 | Per-side mean reviewer rating; `null` if generated |
| `contribution_a`, `contribution_b` | float64 | Per-side mean contribution score; `null` if generated |
| `positive_signals_a/_b` | list[string] | Per-side reviewer excerpts praising originality |
| `negative_signals_a/_b` | list[string] | Per-side reviewer excerpts faulting originality |

Idea order within a pair is shuffled, so a judge cannot do well by always picking a position.


## Citation

```bibtex
TODO: add BibTeX once the preprint is out.
@misc{noveltyjudgebench,
  title  = {Automatic Novelty Judges Are Brittle, Especially on AI-Generated Ideas},
  year   = {2026}
}
```
