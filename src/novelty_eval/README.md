# Novelty Evaluation Benchmarking

### 1. Benchmark Data Preparation (`benchmark_data/`)
Prepare the test set by selecting novel (iclr) and not novel (iclr or generated) ideas.

- `fetch_iclr_data.py` — Downloads ICLR submissions via the OpenReview API, extracts novelty signals and filters by ICLR area rating score percentile.
- `create_benchmark_instances.py` — Assembles the fetched data into benchmark instance YAML files (pairwise or pointwise format), pairing pos-iclr and neg-iclr/generated ideas.
- `paper_blocklist.yaml` — List of papers excluded from instance creation and metric computation due to LLM safety-filter issues.

### 2. Retrieval (`/retrieval`)
Fetch related research papers for each idea to provide the judge with "Related Work" context.
- `retrieve_candidates.py` — Generates LLM-based search queries for each idea, retrieves matching papers from Paper Finder/Semantic Scholar, and maintains a disk-backed cache.
- `flatten_retrieval_cache.py` — Merges flattens a pairwise retrieval caches into a pointwise retrieval cache file.

### 3. Novelty Benchmarking
The core evaluation phase where the judge model (e.g., GPT-5.4) performs the task and results are compared against ground truth.

**Main Files**:
- `run_benchmark.py` — Entry point for experiments; runs a single judge configuration (ranking or pairwise mode) against a config file and produces metric reports.
- `judge.py` — Invokes the LLM judge to compare or pointwise-score ideas, with support for async calls and the OpenAI Batch API.
- `metrics.py` — Computes all evaluation metrics across the three test modes: ranking, pairwise, and pointwise.
- `experiment_stats.py` — Dataclass that collects per-experiment metric lists for clean pass-through between components.
- `tournament.py` — Async tournament runners (round-robin and Swiss) that schedule which idea pairs the judge compares.
- `scoring.py` — Translates raw LLM winner verdicts into Elo-style score increments and aggregates bidirectional comparisons.
- `retrieve_batch_results.py` — Polls and retrieves completed OpenAI Batch API jobs, then reassembles the judge outputs into the standard results format.

**Reporting**:
- `comparison_logger.py` — Writes per-comparison debug entries (prompts, responses, winners) to disk for post-hoc inspection.
- `report_writer.py` — Generates a plain-text summary report of accuracy experiment results.
- `md_report_writer.py` — Generates a detailed Markdown debug report with idea details, paper signals, and per-comparison reasoning.
- `cost_report_writer.py` — Writes a human-readable Markdown cost report (token counts, cache hits, estimated USD) for a benchmark run.

**Benchmark Sweep**:
- `run_benchmark_sweep.py` — Runs `run_benchmark.py` over multiple configurations and produces a cross-config comparison report (config: `config/accuracy_sweep_config.yaml`).

### 4. Ablation Studies (`ablation/`)
Tools for defining, running, and analysing controlled ablation experiments — e.g. varying retrieval, reasoning effort, or judge model.

**Config**:
- `ablations.yaml` — Single source of truth for all ablation experiment definitions (tracks, conditions, descriptions).
- `config.py` — Stateless helpers for loading and resolving ablation configs from `ablations.yaml`.
- `configs/merge/` — Merge configs passed to `merge_ablation_runs.py --config`: which sweep directories to combine, the output dir, and any `baseline:` override. `retrieval_given_backbone/` holds one config per (backbone, setup) pair.

**Execution**:
- `run_ablations.py` — Orchestrator that reads `ablations.yaml`, launches instance creation + accuracy sweeps for each condition, and produces a unified report.
- `batch_manager.py` — State-file I/O and batch-status polling for running ablations asynchronously via the OpenAI Batch API.

**Analysis & Reporting**:
- `merge_ablation_runs.py` — Merges artifact directories from multiple ablation runs into one unified comparison report, and writes the chart-data JSON the figure scripts consume.
- `compare_retrieval_impact.py` — Compares baseline vs. retrieval-augmented runs and generates a Markdown report of where retrieval helped or hurt.
- `derive_subrun_artifacts.py` — Carves independent sub-runs (`mec_k_1`, `unidirectional`) out of a `current` sweep so they can be merged alongside it.

The artifact-reading and metric layer these build on lives in `analysis/`, and the camera-ready figures in `figures/` — neither is ablation-specific.

### 5. Shared Analysis Layer (`analysis/`)
Reads the artifacts a benchmark run writes, filters them, and turns them into metrics. Used by `ablation/`, `run_benchmark_sweep.py`, `retrieval_faceoff/`, `baselines/`, and the notebooks — so it sits below all of them rather than inside any one.

- `artifacts.py` — Discovers artifact directories and extracts per-problem outcomes from `scores.json` and `accuracy_report.txt` files.
- `filtering.py` — Applies the post-hoc paper blocklist (`benchmark_data/paper_blocklist.yaml`), derives which test instances to exclude, and recomputes metrics on the filtered subset.
- `stats_one_pass.py` — Single-pass bootstrap significance testing for pairwise and pointwise metrics.
- `generate_report.py` — Aggregates `accuracy_report.txt` files across conditions into a single Markdown comparison table with heatmaps. Reads the ablation registry at `ablation/ablations.yaml` for canonical row ordering.

### 6. Paper Figures (`figures/`)
Camera-ready figure and table production. These read the `unified_*.json` chart data that `merge_ablation_runs.py` writes, so they redraw without re-running a merge or a bootstrap, and default to writing under `output/paper_figures/`. All but `viz.py` and `unified_chart_data.py` are hand-run CLIs — see each module's docstring for a worked invocation.

- `viz.py` — The heatmap and forest-plot primitives every figure below is built from, plus `_save_figure` (Type-42 fonts, multi-format output).
- `unified_chart_data.py` — On-disk format for the chart data: written by the merge, read back by the rechart CLI. Dependency-light on purpose.
- `chart_two_track.py` — The main two-track figure: one row per ablation, one column per judge, two setups side by side.
- `chart_tie_rates.py` — Judge tie-rate strip plot, read straight from each run's `accuracy_report.txt`. Also writes the CSV the table below consumes.
- `table_tie_rates.py` — The tie-rate numbers as a LaTeX table, from that CSV, so figure and table cannot drift.
- `chart_negatives_source.py` — Holds the ablation fixed and varies where the negatives came from; absolute scores rather than deltas.
- `rechart_unified.py` — Re-renders the unified figures from saved chart data. Sub-second loop for iterating on format.
- `collect_chart_panels.py` — Gathers panels from several merges into one chart-data set, for rows that each need their own baseline.
- `configs/` — Paper wording passed via `--config`: row/column labels, titles, and panel ordering, kept out of argv.
