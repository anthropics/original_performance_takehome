#!/bin/bash
# Update a single candidate's score in scores.txt
# Usage: scripts/update_score.sh <candidate_id> <cycles>
# Updates or adds the score, then re-sorts the file

if [ -z "$2" ]; then
    echo "Usage: $0 <candidate_id> <cycles>"
    exit 1
fi

ID="$1"
CYCLES="$2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SCORES_FILE="$ROOT_DIR/candidates/scores.txt"

# Create scores file if it doesn't exist
touch "$SCORES_FILE"

# Remove old entry for this candidate (if exists)
grep -v "^$ID " "$SCORES_FILE" > "$SCORES_FILE.tmp" || true

# Add new entry
echo "$ID $CYCLES" >> "$SCORES_FILE.tmp"

# Sort by cycles (ascending) and save
sort -t' ' -k2 -n "$SCORES_FILE.tmp" > "$SCORES_FILE"
rm -f "$SCORES_FILE.tmp"

echo "Updated $ID: $CYCLES cycles"
