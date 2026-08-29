#!/usr/bin/env bash
# One-time environment setup for the novelty baselines.
#
# Python 3.12 comes from conda because the cluster default is 3.11.2. The
# environment, the Specter2 weights and every download cache go under the
# project root, not $HOME, which is tightly quota-limited here.
#
# Too long for the GW node, so it needs a compute node. No GPU required.
#
#   srun -c4 --mem-per-cpu=10g --time=2:00:00 --pty $SHELL
#   ./scripts/cluster/setup.sh [--force]      # --force rebuilds the environment

set -euo pipefail

# shellcheck source=scripts/cluster/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REQUIREMENTS="${PROJECT_ROOT}/requirements-baselines.txt"
PYTHON_VERSION="3.12"

[[ -z "${1:-}" || "$1" == "--force" ]] || { echo "Unknown argument: $1" >&2; exit 2; }

require_slurm 'srun -c4 --mem-per-cpu=10g --time=2:00:00 --pty $SHELL'
[[ -f "$REQUIREMENTS" ]] || { echo "ERROR: $REQUIREMENTS not found." >&2; exit 1; }

mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

# sbatch opens its --output file before the job script runs, so this has to
# exist at submission time or the job dies before reaching any code.
mkdir -p "${PROJECT_ROOT}/output/cluster_logs"

load_conda

if [[ "${1:-}" == "--force" && -d "$ENV_PREFIX" ]]; then
  conda env remove --prefix "$ENV_PREFIX" --yes
fi

if [[ -d "$ENV_PREFIX" ]]; then
  echo "Reusing environment at $ENV_PREFIX"
else
  conda create --prefix "$ENV_PREFIX" "python=${PYTHON_VERSION}" --yes
fi

conda activate "$ENV_PREFIX"

ACTUAL_VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$ACTUAL_VERSION" != "$PYTHON_VERSION" ]]; then
  echo "ERROR: environment has Python $ACTUAL_VERSION, expected $PYTHON_VERSION." >&2
  echo "Re-run with --force to recreate it." >&2
  exit 1
fi

echo "Installing requirements (this is the slow part)..."
python -m pip install --upgrade pip setuptools
python -m pip install -r "$REQUIREMENTS"

# Importing the embedding module is the prefetch: it pulls the tokenizer, the
# base model and the specter2 adapter, so the cache holds exactly what the
# benchmark needs at runtime.
echo "Pre-downloading Specter2 weights into $HF_CACHE"
PYTHONPATH="${PROJECT_ROOT}/src" python -c \
  "from novelty_eval.baselines.scideator.ranking.embedding import Specter2Embedding; Specter2Embedding()"

echo
echo "Setup complete. Nothing was written to \$HOME."
echo "  environment : $ENV_PREFIX"
echo "  hf cache    : $HF_CACHE"
echo "  build cache : $CACHE_ROOT (pip wheels and conda tarballs; safe to delete)"
