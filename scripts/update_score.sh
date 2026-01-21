#!/bin/bash
# Update a single candidate's score in scores.txt
# Usage: scripts/update_score.sh <base_dir> <candidate_id> <cycles>
# Updates or adds the score, then re-sorts the file

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <base_dir> <candidate_id> <cycles>"
    exit 1
fi

# Get absolute path for base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_DIR="$PROJECT_ROOT/$1"
ID="$2"
CYCLES="$3"

SCORES_FILE="$BASE_DIR/candidates/scores.txt"

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
