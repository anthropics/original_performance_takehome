#!/bin/bash
# Initialize a candidate folder for optimization
# Usage: scripts/init_candidate.sh <base_dir> <ID>
# Creates candidate folder with baseline code

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <base_dir> <candidate_id>"
    exit 1
fi

# Get absolute path for base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_NAME="$1"
BASE_DIR="$PROJECT_ROOT/$1"
ID="$2"

CANDIDATE_DIR="$BASE_DIR/candidates/CAND_$ID"

mkdir -p "$CANDIDATE_DIR"

# Copy files from project root
cp "$PROJECT_ROOT/perf_takehome.py" "$CANDIDATE_DIR/"
cp "$PROJECT_ROOT/tests/submission_tests.py" "$CANDIDATE_DIR/"

# Determine how many levels up to project root
# From CAND_XXX -> candidates -> {ga,sa} -> project root = 3 levels
sed -i "s|parentdir = os.path.dirname(currentdir)|parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))|" "$CANDIDATE_DIR/submission_tests.py"
sed -i "s|from frozen_problem import|from problem import|" "$CANDIDATE_DIR/submission_tests.py"
sed -i "s|from perf_takehome import|from ${BASE_NAME}.candidates.CAND_$ID.perf_takehome import|" "$CANDIDATE_DIR/submission_tests.py"

# Ensure __init__.py exists
touch "$CANDIDATE_DIR/__init__.py"

echo "Initialized candidate $ID"
