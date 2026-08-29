#!/usr/bin/env bash
# Paths, cache redirection and conda helpers. Sourced, not executed.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_PREFIX="${PROJECT_ROOT}/.conda-baselines"
HF_CACHE="${PROJECT_ROOT}/.hf_cache"
CACHE_ROOT="${PROJECT_ROOT}/.cache"

# Keep every cache off $HOME, which is tightly quota-limited on this cluster.
# XDG_CACHE_HOME covers the tools that honour it; the rest need naming. conda's
# package cache is the easiest to miss — it unpacks gigabytes under $HOME even
# when the environment itself lives elsewhere.
export XDG_CACHE_HOME="${CACHE_ROOT}"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda-pkgs"
export HF_HOME="${HF_CACHE}"
export TORCH_HOME="${CACHE_ROOT}/torch"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"


# Refuse to run on the GW node. Every compute-node context sets SLURM_JOB_ID.
require_slurm() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not inside a SLURM allocation. Work does not run on the GW node." >&2
    echo "Allocate one first:  $1" >&2
    exit 1
  fi
}


# Source conda.sh, so `conda activate` works in a non-login shell.
load_conda() {
  local roots=("$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "${PROJECT_ROOT}/../miniconda3")
  [[ -n "${CONDA_EXE:-}" ]] && roots=("$(dirname "$(dirname "$CONDA_EXE")")" "${roots[@]}")

  local root
  for root in "${roots[@]}"; do
    if [[ -f "${root}/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      . "${root}/etc/profile.d/conda.sh"
      return
    fi
  done

  echo "ERROR: could not find conda.sh. Set CONDA_EXE to your conda binary." >&2
  exit 1
}


# Activate the environment setup.sh built.
activate_env() {
  [[ -d "$ENV_PREFIX" ]] || { echo "ERROR: no environment at $ENV_PREFIX. Run ./scripts/cluster/setup.sh first." >&2; exit 1; }
  conda activate "$ENV_PREFIX"
}
