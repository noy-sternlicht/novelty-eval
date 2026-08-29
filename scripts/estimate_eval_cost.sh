#!/usr/bin/env bash
# estimate_eval_cost.sh — 2-instance cost probes across selected ablations × models.
#
# Edit the MODELS, ABLATIONS, and TRACKS variables below, then run:
#   ./scripts/estimate_eval_cost.sh
#
# Existing test instances are reused (--skip-create). Ablations whose instances
# have never been created are skipped automatically with a warning.
#
# Results land in output/ablation_sweeps/cost_estimation_<timestamp>/ and include
# ablation_cost_summary.md with per-mode costs and full-set extrapolations.
#
# Extra flags are forwarded to run_ablations.py:
#   --parallel-ablations   run all ablations concurrently (faster, noisier logs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src"
export SECRETS="${PROJECT_ROOT}/secrets.toml"
export OUTPUT_DIR="${PROJECT_ROOT}/output"

# ==============================================================================
# ✏️  EDIT HERE
# ==============================================================================

# Judge models to evaluate.
MODELS=(
#    gpt-5.1
#    gpt-5.2
#    gpt-5.4
    claude-sonnet-4-5
#    claude-sonnet-4-6
    claude-opus-4-5
    claude-opus-4-6
)

# Ablations to run.
# Accepts short names (e.g. "current") or full expanded names (e.g. "pairwise_current").
# Short names are expanded to all matching tracks; combine with TRACKS to restrict.
# Leave empty to run ALL ablations registered in ablations.yaml.
ABLATIONS=(
    current
#    vague_criterion
#    unidirectional
#    mec_k_1
#    low_judge_reasoning
#    boring_negatives
#    retrieval
)

# Tracks to restrict to (pairwise, ranking, pointwise).
# Leave empty to run all tracks present in the ABLATIONS list above.
TRACKS=(
     pairwise
#     ranking
     pointwise
)

# Max number of models to run concurrently.
# Anthropic enforces a concurrent-connection limit — keep this at 3–4 when
# the MODELS list contains Anthropic models to avoid 429 rate limit errors.
# Set to 0 (or comment out) to run all models fully in parallel.
MAX_PARALLEL_MODELS=1

# ==============================================================================

# ─── Build comma-separated args ───────────────────────────────────────────────
MODELS_ARG="$(IFS=,; echo "${MODELS[*]}")"

ABLATION_ARGS=()
if [[ "${#ABLATIONS[@]}" -gt 0 ]]; then
    ABLATIONS_ARG="$(IFS=,; echo "${ABLATIONS[*]}")"
    ABLATION_ARGS+=("--ablations" "${ABLATIONS_ARG}")
fi

TRACKS_ARGS=()
if [[ "${#TRACKS[@]}" -gt 0 ]]; then
    TRACKS_ARG="$(IFS=,; echo "${TRACKS[*]}")"
    TRACKS_ARGS+=("--tracks" "${TRACKS_ARG}")
fi

# ─── Output location ──────────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COST_OUTPUT_DIR="${PROJECT_ROOT}/output/ablation_sweeps/cost_estimation_${TIMESTAMP}"

# ─── Summary banner ───────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Evaluation Cost Estimation"
echo " Models    : ${MODELS_ARG}"
echo " Ablations : ${#ABLATIONS[@]} selected"
echo " Instances : x per ablation (full-set cost is extrapolated)"
echo " Output    : ${COST_OUTPUT_DIR}"
echo "================================================================"
echo ""

# ─── Build --max-parallel-models arg ─────────────────────────────────────────
MAX_PARALLEL_ARGS=()
if [[ -n "${MAX_PARALLEL_MODELS:-}" ]] && [[ "${MAX_PARALLEL_MODELS}" -gt 0 ]]; then
    MAX_PARALLEL_ARGS+=("--max-parallel-models" "${MAX_PARALLEL_MODELS}")
fi

# ─── Run ──────────────────────────────────────────────────────────────────────
# --skip-create    : reuse existing instances; skip with warning if none found.
# --parallel-models: run all models for each ablation concurrently (capped by MAX_PARALLEL_MODELS).
python3 "${PROJECT_ROOT}/src/novelty_eval/ablation/run_ablations.py" \
    --models "${MODELS_ARG}" \
    --num-instances 5 \
    --skip-create \
    --parallel-models \
    --output-dir "${COST_OUTPUT_DIR}" \
    ${ABLATION_ARGS[@]+"${ABLATION_ARGS[@]}"} \
    ${TRACKS_ARGS[@]+"${TRACKS_ARGS[@]}"} \
    ${MAX_PARALLEL_ARGS[@]+"${MAX_PARALLEL_ARGS[@]}"} \
    "$@"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Cost estimation complete."
echo ""
echo " Human-readable summary:"
echo "   ${COST_OUTPUT_DIR}/ablation_cost_summary.md"
echo ""
echo " Raw data:"
echo "   ${COST_OUTPUT_DIR}/ablation_cost_summary.json"
echo "================================================================"
