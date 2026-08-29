#!/usr/bin/env bash
# Run the Scideator novelty baseline on a compute node. Already inside an
# allocation, so it works the same under srun and under sbatch_run.sh.
#
#   ./scripts/cluster/run.sh [--set KEY=VALUE ...] [--config PATH]
#
# Arguments go to run_benchmark.py; later flags win, so --config and --set
# override the defaults below.

set -euo pipefail

# shellcheck source=scripts/cluster/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_slurm 'srun --gres=gg:g4:1 --exclude=wadi-01,wadi-02,wadi-03,wadi-04,wadi-05 --time=2:00:00 --mem-per-cpu=10g --pty $SHELL'

load_conda
activate_env

export PYTHONPATH="${PROJECT_ROOT}/src"
export SECRETS="${PROJECT_ROOT}/secrets.toml"
export OUTPUT_DIR="${PROJECT_ROOT}/output"
mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

# The benchmark resolves test_inputs relative to the project root.
cd "$PROJECT_ROOT"

CONFIG="src/novelty_eval/config/accuracy_test_scideator.yaml"

python scripts/cluster/preflight.py --config "$CONFIG" "$@"

# The checked-in config carries a laptop output_dir; --set replaces it here and
# the effective config is saved with the run's artifacts.
exec python src/novelty_eval/run_benchmark.py \
  --config "$CONFIG" --set "output_dir=${OUTPUT_DIR}" "$@"
