#!/bin/bash
# Select top N candidates and delete the rest
# Usage: scripts/select_survivors.sh <n>
# Output: Lists kept and deleted candidates
# Requires: candidates/scores.txt must exist

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <population_size>"
    exit 1
fi

N="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SCORES_FILE="$ROOT_DIR/candidates/scores.txt"
CANDIDATES_DIR="$ROOT_DIR/candidates"

if [ ! -f "$SCORES_FILE" ]; then
    echo "ERROR: scores.txt not found. Run evaluations first."
    exit 1
fi

# Get all candidates sorted by score (ascending = better)
ALL_CANDIDATES=$(sort -t' ' -k2 -n "$SCORES_FILE" | awk '{print $1}')
TOTAL=$(echo "$ALL_CANDIDATES" | wc -l)

if [ "$TOTAL" -le "$N" ]; then
    echo "KEPT: $ALL_CANDIDATES"
    echo "DELETED: (none - population already at or below target size)"
    exit 0
fi

# Get survivors (top N)
SURVIVORS=$(echo "$ALL_CANDIDATES" | head -n "$N")

# Get eliminated (rest)
ELIMINATED=$(echo "$ALL_CANDIDATES" | tail -n +$((N + 1)))

# Delete eliminated candidates
for ID in $ELIMINATED; do
    rm -rf "$CANDIDATES_DIR/CAND_$ID"
    # Remove from scores.txt
    grep -v "^$ID " "$SCORES_FILE" > "$SCORES_FILE.tmp" || true
    mv "$SCORES_FILE.tmp" "$SCORES_FILE"
done

echo "KEPT: $(echo $SURVIVORS | tr '\n' ' ')"
echo "DELETED: $(echo $ELIMINATED | tr '\n' ' ')"
