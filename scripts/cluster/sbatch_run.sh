#!/bin/bash -x
#SBATCH --job-name=scideator
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gg:g4:1
#SBATCH --exclude=wadi-01,wadi-02,wadi-03,wadi-04,wadi-05
#SBATCH --output=output/cluster_logs/slurm-%j.out
#SBATCH --error=output/cluster_logs/slurm-%j.err

# Batch submission. Submit from the project root on the GW node:
#
#   sbatch scripts/cluster/sbatch_run.sh
#   sbatch --time=2:00:00 scripts/cluster/sbatch_run.sh --set num_instances=5
#
# #SBATCH lines are read as plain text before anything runs, so a variable
# there would never expand. Override on the command line instead, where sbatch
# flags take precedence. Arguments after the script name go to run.sh.
#
# Use this rather than srun for the full 313 instances: an interactive session
# dies with the SSH connection, which a multi-day run will not survive.
#
# Watch it:  tail -f output/cluster_logs/slurm-<job-id>.out
# Stop it:   scancel <job-id>

set -euo pipefail

# sbatch may run this from a spool copy, so BASH_SOURCE is not a reliable way
# back to the project. SLURM_SUBMIT_DIR is where the job was submitted from.
PROJECT_ROOT="${SCIDEATOR_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [[ ! -x "${PROJECT_ROOT}/scripts/cluster/run.sh" ]]; then
  echo "ERROR: no project at ${PROJECT_ROOT}. Submit from the project root," >&2
  echo "or set SCIDEATOR_PROJECT_ROOT." >&2
  exit 1
fi

cd "$PROJECT_ROOT"

exec ./scripts/cluster/run.sh "$@"
