#!/bin/bash
# Evaluate a candidate and return cycle count
# Usage: scripts/eval_candidate.sh <candidate_id>
# Output: cycle count (integer) or "ERROR" on failure

if [ -z "$1" ]; then
    echo "Usage: $0 <candidate_id>"
    exit 1
fi

ID="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CANDIDATE_DIR="$ROOT_DIR/candidates/CAND_$ID"

if [ ! -d "$CANDIDATE_DIR" ]; then
    echo "ERROR: Candidate $ID does not exist"
    exit 1
fi

cd "$CANDIDATE_DIR"
# Run tests, capture output (ignore exit code - tests may fail but still report cycles)
RESULT=$(python submission_tests.py 2>&1 || true)

# Extract cycle count from output (first CYCLES line)
CYCLES=$(echo "$RESULT" | grep "CYCLES:" | head -1 | awk '{print $2}')

if [ -z "$CYCLES" ]; then
    echo "ERROR"
    exit 1
fi

echo "$CYCLES"
