# Running the Scideator baseline on the cluster

Project root on the cluster: `/sci/labs/tomhope/noystl/scideator_baseline`

**Never run work on the GW node.** It only gets `sbatch`, `squeue`, `scancel`.

## 1. Push (from your laptop)

```bash
export SCIDEATOR_REMOTE=noystl@phoenix-gw.cs.huji.ac.il
python scripts/cluster/cluster.py push
```

Sends code, `secrets.toml`, and the config's `test_inputs`. Code is chosen by
`git ls-files`, so gitignored results never ship. `--dry-run` to preview.

## 2. Set up the environment (once)

```bash
srun -c4 --mem-per-cpu=10g --time=2:00:00 --pty $SHELL
cd /sci/labs/tomhope/noystl/scideator_baseline
./scripts/cluster/setup.sh
```

Builds `.conda-baselines/` (Python 3.12), installs `requirements-baselines.txt`,
prefetches Specter2. Slow — it is the whole torch stack. `--force` to rebuild.

## 3. Run

```bash
sbatch --time=2:00:00 scripts/cluster/sbatch_run.sh --set num_instances=5   # smoke
sbatch scripts/cluster/sbatch_run.sh                                        # full
tail -f output/cluster_logs/slurm-<job-id>.out
```

`sbatch` flags override the `#SBATCH` defaults; arguments after the script name
go to `run_benchmark.py` (`--config PATH`, `--set KEY=VALUE`).

Two jobs at once, differing only in the judge model:

```bash
sbatch scripts/cluster/sbatch_run.sh
sbatch scripts/cluster/sbatch_run.sh \
  --set llm_engine=claude-opus-4-6 --set anthropic_max_workers=1
```

Override on the sbatch line rather than editing `run.sh` or a config in place.
`sbatch` spools a copy of `sbatch_run.sh` at submission, but `run.sh` and the
YAML are read when the job *starts* — so an edit made while the first job is
still queued silently changes what it runs.

To watch it live instead, take an allocation and run the work script directly:

```bash
srun --gres=gg:g4:1 --exclude=wadi-01,wadi-02,wadi-03,wadi-04,wadi-05 \
     --time=2:00:00 --mem-per-cpu=10g --pty $SHELL
./scripts/cluster/run.sh --set num_instances=5
```

## 4. Pull (from your laptop)

```bash
python scripts/cluster/cluster.py pull                    # newest run, plus slurm logs
python scripts/cluster/cluster.py runs                    # what is available
python scripts/cluster/cluster.py pull --run <timestamp>
```

## Notes

- GPU flag is `gg:g4:1`, not `gpu:...`. `wadi-01..05` have no g4 cards.
- The GPU only serves Specter2. Everything else is API calls and Semantic
  Scholar at ~1 RPS, so the run is network-bound and a missing GPU is a
  warning, not an error.
- **No resume.** `update_report` re-scores saved artifacts without calling an
  LLM; `num_instances` takes the first N with no offset. A killed job re-runs
  from instance 1 and re-pays for completed work. Hence the 6-day default.
- Partitions may cap `--time`. Check `sinfo -o "%P %l"`.
- Caches are kept off `$HOME`, which is quota-limited: environment, Specter2
  weights, and the pip/conda/torch caches all live under the project root.
- The checked-in config carries a laptop `output_dir`; `run.sh` overrides it
  with `--set`. Each run saves its `effective_config.yaml` alongside results.
- A `claude-*` `llm_engine` makes `run_benchmark.py` discard `max_workers` and
  use `anthropic_max_workers` (default 3) instead. Set it to match, or the
  Claude run goes wider against Semantic Scholar than the config asks for.
- Run directories are bare start-time timestamps, so parallel jobs are told
  apart by the `llm_engine` in their `effective_config.yaml`.

## Useful

```bash
ssinfo                 # available GPU types
ssqueue -u noystl      # your jobs
scancel <job-id>
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,NodeList
```
