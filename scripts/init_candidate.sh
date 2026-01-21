#!/bin/bash
# Initialize a candidate folder for genetic algorithm optimization
# Usage: scripts/init_candidate.sh <ID>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <candidate_id>"
    exit 1
fi

ID="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CANDIDATE_DIR="$ROOT_DIR/candidates/CAND_$ID"

mkdir -p "$CANDIDATE_DIR"

# Copy files
cp "$ROOT_DIR/perf_takehome.py" "$CANDIDATE_DIR/"
cp "$ROOT_DIR/tests/submission_tests.py" "$CANDIDATE_DIR/"

# Fix imports in test file
sed -i "s|parentdir = os.path.dirname(currentdir)|parentdir = os.path.dirname(os.path.dirname(currentdir))|" "$CANDIDATE_DIR/submission_tests.py"
sed -i "s|from frozen_problem import|from problem import|" "$CANDIDATE_DIR/submission_tests.py"
sed -i "s|from perf_takehome import|from candidates.CAND_$ID.perf_takehome import|" "$CANDIDATE_DIR/submission_tests.py"

echo "Initialized candidate $ID"
