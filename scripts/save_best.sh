#!/bin/bash
# Save a candidate as the current best
# Usage: scripts/save_best.sh <candidate_id>
# Copies candidate to best/ folder, keeps only perf_takehome.py

if [ -z "$1" ]; then
    echo "Usage: $0 <candidate_id>"
    exit 1
fi

ID="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CANDIDATE_DIR="$ROOT_DIR/candidates/CAND_$ID"
BEST_DIR="$ROOT_DIR/best"

if [ ! -d "$CANDIDATE_DIR" ]; then
    echo "ERROR: Candidate $ID does not exist"
    exit 1
fi

# Remove old best if exists
rm -rf "$BEST_DIR"

# Create best directory and copy only perf_takehome.py
mkdir -p "$BEST_DIR"
cp "$CANDIDATE_DIR/perf_takehome.py" "$BEST_DIR/"

echo "Saved candidate $ID as best"
