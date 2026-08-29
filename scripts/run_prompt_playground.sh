#!/bin/bash
# Run the Comparison Prompt Playground from the project root
#
# Usage:
#   ./scripts/run_prompt_playground.sh [OPTIONS]
#
# Options:
#   --input, -i       Input YAML file (default: playground_test_inputs.yaml)
#   --model, -m       LLM model name (e.g., gpt-4o, gpt-5.2, o3)
#   --num-instances   Number of instances to process (default: 1)
#   --output          Output directory (default: playground_results)
#   --full-criteria   Use full evaluation criteria
#
# Examples:
#   ./scripts/run_prompt_playground.sh --model gpt-4o --num-instances 2
#   ./scripts/run_prompt_playground.sh --input test_inputs.yaml --model gpt-5.2 --num-instances 5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Export secrets path
export SECRETS="$PROJECT_ROOT/secrets.toml"

# Default values
INPUT_FILE="playground_test_inputs.yaml"
MODEL="gpt-4o"
NUM_INSTANCES=1
OUTPUT_DIR="playground_results"
FULL_CRITERIA=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --num-instances|-n)
            NUM_INSTANCES="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --full-criteria)
            FULL_CRITERIA="--full-criteria"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "  Comparison Prompt Playground"
echo "========================================"
echo "Input File:     $INPUT_FILE"
echo "Model:          $MODEL"
echo "Num Instances:  $NUM_INSTANCES"
echo "Output Dir:     $OUTPUT_DIR"
echo "Full Criteria:  ${FULL_CRITERIA:-no}"
echo "========================================"
echo ""

cd "$PROJECT_ROOT/src/novelty_eval"

python3 prompt_playground.py \
    --input "$INPUT_FILE" \
    --model "$MODEL" \
    --num-instances "$NUM_INSTANCES" \
    --output "$OUTPUT_DIR" \
    $FULL_CRITERIA

echo ""
echo "========================================"
echo "  Done! Results saved to:"
echo "  $PROJECT_ROOT/src/novelty_eval/$OUTPUT_DIR"
echo "========================================"

