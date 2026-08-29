#!/bin/bash -x
#SBATCH --time=1:00:00
#SBATCH -c1
#SBATCH --mem-per-cpu=1g

set_env_vars() {
  export PYTHONPATH=$PWD/src
  export SECRETS=$PWD/secrets.toml
  export OUTPUT_DIR=$PWD/output
}

source "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate "$PWD"/myenv
set_env_vars

python3 src/novelty_eval/run_benchmark.py \
  --config src/novelty_eval/config.toml \
  --test-inputs src/novelty_eval/iclr_test_instances.yaml\
  --output-file accuracy_report.txt \
  --max-workers 30 \
  -n 3