#!/usr/bin/env python3
"""Checks run before the benchmark, so a bad allocation or a missing file fails
now rather than after twenty minutes of Semantic Scholar calls."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def fail(message: str, hint: str = "") -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    sys.exit(1)


def check_files(config: Path) -> None:
    if not (ROOT / "secrets.toml").exists():
        fail(f"{ROOT}/secrets.toml not found.",
             "It is gitignored, so it has to be copied to the cluster separately.")
    if not config.exists():
        fail(f"config not found: {config}")

    test_inputs = yaml.safe_load(config.read_text()).get("test_inputs")
    if test_inputs and not (ROOT / test_inputs).exists():
        fail(f"test_inputs not found: {test_inputs}",
             "It lives under output/, which is gitignored — copy it up separately.")


def report_hardware() -> None:
    """The GPU only serves Specter2 embedding; everything else is network-bound
    API calls, so a missing GPU is a warning rather than an error."""
    print("--- hardware ---")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"], check=False)
    else:
        print("nvidia-smi not on PATH")

    import torch
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("\nWARNING: torch cannot see a GPU. Specter2 will embed on CPU.")
        print("Results are still correct, but check that --gres gg:g4:1 was granted.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args()  # shares a command line with run_benchmark.py

    check_files(args.config)
    report_hardware()


if __name__ == "__main__":
    main()
