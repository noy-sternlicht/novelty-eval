#!/bin/bash
# run_accuracy_test_units.sh — Run unit and integration tests for run_benchmark.py

# Determine project root (one level up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set up environment
export PYTHONPATH="$PROJECT_ROOT/src"
export OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/logs/test_logs"}
mkdir -p "$OUTPUT_DIR"

echo "Running run_benchmark unit and integration tests..."

# Execute from project root to ensure file paths are resolved correctly
cd "$PROJECT_ROOT"
./.venv/bin/python3 -m unittest tests/test_run_benchmark.py

