#!/bin/bash
# Get statistics about the current population
# Usage: scripts/get_stats.sh [baseline_cycles]
# Output:
#   BEST: id cycles
#   AVG: cycles
#   IMPROVEMENT: percentage (if baseline provided)

BASELINE="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SCORES_FILE="$ROOT_DIR/candidates/scores.txt"

if [ ! -f "$SCORES_FILE" ]; then
    echo "ERROR: scores.txt not found"
    exit 1
fi

# Get best (first line after sorting)
BEST_LINE=$(head -1 "$SCORES_FILE")
BEST_ID=$(echo "$BEST_LINE" | awk '{print $1}')
BEST_CYCLES=$(echo "$BEST_LINE" | awk '{print $2}')

# Calculate average
AVG=$(awk '{sum += $2; count++} END {printf "%.0f", sum/count}' "$SCORES_FILE")

echo "BEST: $BEST_ID $BEST_CYCLES"
echo "AVG: $AVG"

# Calculate improvement if baseline provided
if [ -n "$BASELINE" ]; then
    IMPROVEMENT=$(python3 -c "print(f'{(1 - $BEST_CYCLES / $BASELINE) * 100:.1f}')")
    echo "IMPROVEMENT: $IMPROVEMENT%"
fi
