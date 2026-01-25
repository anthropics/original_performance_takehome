#!/bin/bash
# Get non-elite candidates (to be replaced via crossover/mutation)
# Usage: scripts/get_non_elite.sh <elite_count>
# Output: candidate IDs, one per line
# Requires: candidates/scores.txt must exist (run eval_all.sh first)

if [ -z "$1" ]; then
    echo "Usage: $0 <elite_count>"
    exit 1
fi

ELITE_COUNT="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SCORES_FILE="$ROOT_DIR/candidates/scores.txt"

if [ ! -f "$SCORES_FILE" ]; then
    echo "ERROR: scores.txt not found. Run eval_all.sh first."
    exit 1
fi

# Sort by cycles (ascending), skip top N elites, output IDs only
sort -t' ' -k2 -n "$SCORES_FILE" | tail -n +$((ELITE_COUNT + 1)) | awk '{print $1}'
