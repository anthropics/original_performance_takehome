#!/bin/bash
# Accept the neighbor as the new current solution
# Usage: sa/scripts/accept_neighbor.sh <neighbor_score>
# Replaces CURRENT with NEIGHBOR

if [ -z "$1" ]; then
    echo "Usage: $0 <neighbor_score>"
    exit 1
fi

NEIGHBOR_SCORE="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
CURRENT_DIR="$SA_DIR/candidates/CAND_CURRENT"
NEIGHBOR_DIR="$SA_DIR/candidates/CAND_NEIGHBOR"
STATE_FILE="$SA_DIR/candidates/state.txt"

if [ ! -d "$NEIGHBOR_DIR" ]; then
    echo "ERROR: NEIGHBOR does not exist"
    exit 1
fi

# Remove old CURRENT and move NEIGHBOR to CURRENT
rm -rf "$CURRENT_DIR"
mv "$NEIGHBOR_DIR" "$CURRENT_DIR"

# Update import path in submission_tests.py
sed -i "s|from sa.candidates.CAND_NEIGHBOR.perf_takehome import|from sa.candidates.CAND_CURRENT.perf_takehome import|" "$CURRENT_DIR/submission_tests.py"

# Update state
sed -i "s|^CURRENT_SCORE=.*|CURRENT_SCORE=$NEIGHBOR_SCORE|" "$STATE_FILE"

# Increment accepted count
ACCEPTED=$(grep "^ACCEPTED_COUNT=" "$STATE_FILE" | cut -d'=' -f2)
NEW_ACCEPTED=$((ACCEPTED + 1))
sed -i "s|^ACCEPTED_COUNT=.*|ACCEPTED_COUNT=$NEW_ACCEPTED|" "$STATE_FILE"

echo "Accepted neighbor with score $NEIGHBOR_SCORE"
