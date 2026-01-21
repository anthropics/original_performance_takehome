#!/bin/bash
# Log an iteration to the history file
# Usage: sa/scripts/log_iteration.sh <iteration> <temperature> <current_score> <best_score> <accepted|rejected>

if [ -z "$5" ]; then
    echo "Usage: $0 <iteration> <temperature> <current_score> <best_score> <accepted|rejected>"
    exit 1
fi

ITERATION="$1"
TEMPERATURE="$2"
CURRENT_SCORE="$3"
BEST_SCORE="$4"
STATUS="$5"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
HISTORY_FILE="$SA_DIR/candidates/history.txt"

# Append to history
echo "$ITERATION $TEMPERATURE $CURRENT_SCORE $BEST_SCORE $STATUS" >> "$HISTORY_FILE"

echo "Logged iteration $ITERATION"
