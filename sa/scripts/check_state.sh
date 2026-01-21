#!/bin/bash
# Check if SA state exists and return current state
# Usage: sa/scripts/check_state.sh
# Output: State values if exists, or "NO_STATE" if fresh start
#
# Exit codes:
#   0 - State exists (can resume)
#   1 - No state (fresh start needed)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
STATE_FILE="$SA_DIR/candidates/state.txt"
CURRENT_DIR="$SA_DIR/candidates/CAND_CURRENT"
BEST_DIR="$SA_DIR/candidates/CAND_BEST"

# Check if all required components exist
if [ -f "$STATE_FILE" ] && [ -d "$CURRENT_DIR" ] && [ -d "$BEST_DIR" ]; then
    # State exists - output current values
    echo "STATE_EXISTS=true"
    cat "$STATE_FILE"
    exit 0
else
    echo "NO_STATE"
    exit 1
fi
