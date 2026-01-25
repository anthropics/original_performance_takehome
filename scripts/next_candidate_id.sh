#!/bin/bash
# Get the next available candidate ID
# Usage: scripts/next_candidate_id.sh <base_dir>
# Output: Next available ID (e.g., "011" if 001-010 exist)

if [ -z "$1" ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

# Get absolute path for base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_DIR="$PROJECT_ROOT/$1"
CANDIDATES_DIR="$BASE_DIR/candidates"

# Find the highest existing ID
MAX_ID=0
if [ -d "$CANDIDATES_DIR" ]; then
    for dir in "$CANDIDATES_DIR"/CAND_*; do
        if [ -d "$dir" ]; then
            # Extract numeric ID, removing leading zeros for comparison
            ID=$(basename "$dir" | sed 's/CAND_//' | sed 's/^0*//')
            if [ -z "$ID" ]; then
                ID=0
            fi
            if [ "$ID" -gt "$MAX_ID" ]; then
                MAX_ID="$ID"
            fi
        fi
    done
fi

# Return next ID with zero-padding
NEXT_ID=$((MAX_ID + 1))
printf "%03d\n" "$NEXT_ID"
