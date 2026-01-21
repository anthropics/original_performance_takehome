#!/bin/bash
# Update best solution if neighbor is better than all-time best
# Usage: sa/scripts/update_best.sh <neighbor_score>
#
# IMPORTANT: Call this BEFORE accept/reject decision!
# We save the best solution we've ever seen, even if we reject it as CURRENT.
# This ensures we don't lose a good solution due to Metropolis rejection.

if [ -z "$1" ]; then
    echo "Usage: $0 <neighbor_score>"
    exit 1
fi

NEIGHBOR_SCORE="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
NEIGHBOR_DIR="$SA_DIR/candidates/CAND_NEIGHBOR"
BEST_DIR="$SA_DIR/candidates/CAND_BEST"
STATE_FILE="$SA_DIR/candidates/state.txt"

if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: state.txt not found"
    exit 1
fi

if [ ! -d "$NEIGHBOR_DIR" ]; then
    echo "ERROR: NEIGHBOR does not exist"
    exit 1
fi

BEST_SCORE=$(grep "^BEST_SCORE=" "$STATE_FILE" | cut -d'=' -f2)

# Check if neighbor is better than all-time best (lower is better)
if python3 -c "exit(0 if $NEIGHBOR_SCORE < $BEST_SCORE else 1)"; then
    # New best found - save NEIGHBOR to BEST
    rm -rf "$BEST_DIR"
    cp -r "$NEIGHBOR_DIR" "$BEST_DIR"

    # Update import path
    sed -i "s|from sa.candidates.CAND_NEIGHBOR.perf_takehome import|from sa.candidates.CAND_BEST.perf_takehome import|" "$BEST_DIR/submission_tests.py"

    # Update state
    sed -i "s|^BEST_SCORE=.*|BEST_SCORE=$NEIGHBOR_SCORE|" "$STATE_FILE"

    echo "NEW_BEST: $NEIGHBOR_SCORE (was $BEST_SCORE)"
else
    echo "NO_CHANGE: $NEIGHBOR_SCORE >= $BEST_SCORE"
fi
