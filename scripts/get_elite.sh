#!/bin/bash
# Get top N elite candidates (lowest cycle counts)
# Usage: scripts/get_elite.sh <n>
# Output: candidate IDs, one per line
# Requires: candidates/scores.txt must exist (run eval_all.sh first)

if [ -z "$1" ]; then
    echo "Usage: $0 <n>"
    exit 1
fi

N="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SCORES_FILE="$ROOT_DIR/candidates/scores.txt"

if [ ! -f "$SCORES_FILE" ]; then
    echo "ERROR: scores.txt not found. Run eval_all.sh first."
    exit 1
fi

# Sort by cycles (ascending), take top N, output IDs only
sort -t' ' -k2 -n "$SCORES_FILE" | head -n "$N" | awk '{print $1}'
