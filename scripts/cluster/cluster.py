#!/usr/bin/env python3
"""Move the project to the cluster and results back. Runs on your laptop.

    cluster.py push          # code, secrets.toml, the config's test_inputs
    cluster.py runs          # what is on the cluster
    cluster.py pull          # newest run's artifacts and slurm logs
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "src/novelty_eval/config/accuracy_test_scideator.yaml"
DEFAULT_REMOTE_PATH = "/sci/labs/tomhope/noystl/scideator_baseline"
ARTIFACTS = "output/accuracy_test_artifacts"
LOGS = "output/cluster_logs"

# Runs are timestamp-named. Filtering on that shape keeps strays out of the
# listing, where "newest" is just the last name in sort order.
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def rsync(args, src: str, dst: str, extra=(), check=True, **kwargs):
    """Flags stay minimal: macOS ships openrsync, not rsync 3.x."""
    cmd = ["rsync", "-az", "--stats", *extra]
    if args.dry_run:
        cmd.append("--dry-run")
    return subprocess.run([*cmd, src, dst], check=check, **kwargs)


def remote_runs(args) -> list:
    listing = subprocess.run(
        ["ssh", args.remote, f"ls -1 {args.path}/{ARTIFACTS} 2>/dev/null"],
        capture_output=True, text=True,
    ).stdout
    return sorted(name for name in listing.split() if RUN_RE.match(name))


def push(args) -> None:
    dest = f"{args.remote}:{args.path}/"

    # git picks the files, so uncommitted work ships and gitignored results
    # never do. --files-from lists files only, so nothing recurses.
    print(f"--- code -> {dest}")
    files = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, check=True,
    ).stdout
    rsync(args, f"{ROOT}/", dest, extra=["--files-from=-", "--from0"], input=files)

    if not (ROOT / "secrets.toml").exists():
        fail("no secrets.toml locally. It is gitignored, so it needs its own transfer.")
    print(f"--- secrets.toml -> {dest}")
    rsync(args, str(ROOT / "secrets.toml"), dest)
    if not args.dry_run:
        # Lab storage is shared; do not trust the cluster's umask.
        subprocess.run(["ssh", args.remote, f"chmod 600 {args.path}/secrets.toml"], check=True)

    test_inputs = yaml.safe_load(args.config.read_text()).get("test_inputs")
    if not test_inputs:
        fail(f"no test_inputs in {args.config.name}")
    if not (ROOT / test_inputs).exists():
        fail(f"test_inputs missing locally: {test_inputs}")
    print(f"--- {test_inputs} -> {dest}")
    # -R replays the relative path remotely, so the config resolves it
    # identically on both machines.
    rsync(args, f"./{test_inputs}", dest, extra=["-R"], cwd=ROOT)

    # sbatch opens its --output file before the job script runs, so this has to
    # exist at submission time or the job dies before reaching any code.
    print(f"--- mkdir {LOGS}")
    if not args.dry_run:
        subprocess.run(["ssh", args.remote, f"mkdir -p {args.path}/{LOGS}"], check=True)

    print("\nDry run — nothing sent." if args.dry_run
          else "\nNext: sbatch scripts/cluster/sbatch_run.sh   (on the GW node)")


def pull(args) -> None:
    runs = remote_runs(args)
    if not runs:
        fail(f"no runs in {args.remote}:{args.path}/{ARTIFACTS}")

    if args.run and args.run not in runs:
        fail(f"no such run: {args.run}. Use `cluster.py runs` to list them.")
    run = args.run or runs[-1]

    print(f"--- {run}")
    (ROOT / ARTIFACTS / run).mkdir(parents=True, exist_ok=True)
    rsync(args, f"{args.remote}:{args.path}/{ARTIFACTS}/{run}/", f"{ROOT / ARTIFACTS / run}/")

    # Only sbatch writes these; an interactive run streams to the terminal
    # instead, so the remote directory may not exist at all.
    print("--- slurm logs")
    (ROOT / LOGS).mkdir(parents=True, exist_ok=True)
    if rsync(args, f"{args.remote}:{args.path}/{LOGS}/", f"{ROOT / LOGS}/", check=False).returncode:
        print("  (none)")

    print("\nDry run — nothing fetched." if args.dry_run else f"\nResults in {ROOT / ARTIFACTS}/")


def list_runs(args) -> None:
    runs = remote_runs(args)
    print("\n".join(runs) if runs else "(none)")


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--remote", default=os.environ.get("SCIDEATOR_REMOTE"),
                        help="ssh target (default: $SCIDEATOR_REMOTE)")
    common.add_argument("--path", default=os.environ.get("SCIDEATOR_REMOTE_PATH", DEFAULT_REMOTE_PATH),
                        help="remote project root")
    common.add_argument("--dry-run", action="store_true", help="show what would transfer")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("push", parents=[common], help="send code, secrets and data up")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="config whose test_inputs to push")
    p.set_defaults(func=push)

    p = sub.add_parser("pull", parents=[common], help="fetch results down")
    p.add_argument("--run", help="run timestamp (default: the newest)")
    p.set_defaults(func=pull)

    p = sub.add_parser("runs", parents=[common], help="list the runs on the cluster")
    p.set_defaults(func=list_runs)

    args = parser.parse_args()
    if not args.remote:
        fail("no remote host. Pass --remote HOST, or export SCIDEATOR_REMOTE=noystl@<gw-host>")
    if os.environ.get("SLURM_JOB_ID"):
        fail("this moves files between your laptop and the cluster, but it is running on a compute node.")
    args.func(args)


if __name__ == "__main__":
    main()
