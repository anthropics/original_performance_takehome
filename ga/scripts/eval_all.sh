#!/bin/bash
# Evaluate all candidates and output sorted ranking
# Usage: scripts/eval_all.sh
# Output: "{id} {cycles}" per line, sorted ascending by cycles
# Also saves to candidates/scores.txt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"
CANDIDATES_DIR="$GA_DIR/candidates"
SCORES_FILE="$CANDIDATES_DIR/scores.txt"

# Clear scores file
> "$SCORES_FILE"

# Find all candidate directories
for dir in "$CANDIDATES_DIR"/CAND_*; do
    if [ -d "$dir" ]; then
        ID=$(basename "$dir" | sed 's/CAND_//')
        CYCLES=$("$PROJECT_ROOT/scripts/eval_candidate.sh" ga "$ID")
        if [ "$CYCLES" != "ERROR" ]; then
            echo "$ID $CYCLES" >> "$SCORES_FILE"
        fi
    fi
done

# Sort by cycles (ascending) and output
sort -t' ' -k2 -n "$SCORES_FILE"
