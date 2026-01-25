#!/bin/bash
# Get statistics about the current SA optimization state
# Usage: sa/scripts/get_stats.sh [baseline_cycles]
# Output:
#   CURRENT: cycles
#   BEST: cycles
#   TEMPERATURE: value
#   ITERATION: count
#   ACCEPTANCE_RATE: percentage
#   IMPROVEMENT: percentage (if baseline provided)

BASELINE="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
STATE_FILE="$SA_DIR/candidates/state.txt"

if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: state.txt not found"
    exit 1
fi

# Read state values
TEMPERATURE=$(grep "^TEMPERATURE=" "$STATE_FILE" | cut -d'=' -f2)
ITERATION=$(grep "^ITERATION=" "$STATE_FILE" | cut -d'=' -f2)
CURRENT_SCORE=$(grep "^CURRENT_SCORE=" "$STATE_FILE" | cut -d'=' -f2)
BEST_SCORE=$(grep "^BEST_SCORE=" "$STATE_FILE" | cut -d'=' -f2)
ACCEPTED=$(grep "^ACCEPTED_COUNT=" "$STATE_FILE" | cut -d'=' -f2)
REJECTED=$(grep "^REJECTED_COUNT=" "$STATE_FILE" | cut -d'=' -f2)

echo "CURRENT: $CURRENT_SCORE"
echo "BEST: $BEST_SCORE"
echo "TEMPERATURE: $TEMPERATURE"
echo "ITERATION: $ITERATION"

# Calculate acceptance rate
TOTAL=$((ACCEPTED + REJECTED))
if [ "$TOTAL" -gt 0 ]; then
    RATE=$(python3 -c "print(f'{$ACCEPTED / $TOTAL * 100:.1f}')")
    echo "ACCEPTANCE_RATE: $RATE%"
else
    echo "ACCEPTANCE_RATE: N/A"
fi

# Calculate improvement if baseline provided
if [ -n "$BASELINE" ]; then
    IMPROVEMENT=$(python3 -c "print(f'{(1 - $BEST_SCORE / $BASELINE) * 100:.1f}')")
    echo "IMPROVEMENT: $IMPROVEMENT%"
fi
