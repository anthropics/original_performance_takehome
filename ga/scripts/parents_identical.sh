#!/bin/bash
# Check if two parent candidates have identical perf_takehome.py files
# Usage: scripts/parents_identical.sh <parent1_id> <parent2_id>
# Returns: exit code 0 if identical, 1 if different

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <parent1_id> <parent2_id>"
    exit 1
fi

PARENT1="$1"
PARENT2="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

FILE1="$ROOT_DIR/candidates/CAND_$PARENT1/perf_takehome.py"
FILE2="$ROOT_DIR/candidates/CAND_$PARENT2/perf_takehome.py"

if [ ! -f "$FILE1" ]; then
    echo "ERROR: $FILE1 not found"
    exit 1
fi

if [ ! -f "$FILE2" ]; then
    echo "ERROR: $FILE2 not found"
    exit 1
fi

# Compare files (ignoring whitespace differences)
if diff -q "$FILE1" "$FILE2" > /dev/null 2>&1; then
    exit 0  # Identical
else
    exit 1  # Different
fi
