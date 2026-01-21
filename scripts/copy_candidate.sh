#!/bin/bash
# Copy a candidate folder to a new ID
# Usage: scripts/copy_candidate.sh <source_id> <dest_id>
# Output: "Copied {source} to {dest}"

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <source_id> <dest_id>"
    exit 1
fi

SOURCE_ID="$1"
DEST_ID="$2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$ROOT_DIR/candidates/CAND_$SOURCE_ID"
DEST_DIR="$ROOT_DIR/candidates/CAND_$DEST_ID"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source candidate $SOURCE_ID does not exist"
    exit 1
fi

if [ -d "$DEST_DIR" ]; then
    echo "ERROR: Destination candidate $DEST_ID already exists"
    exit 1
fi

# Copy the directory
cp -r "$SOURCE_DIR" "$DEST_DIR"

# Update the import in submission_tests.py to point to the new candidate
sed -i "s|from candidates.CAND_$SOURCE_ID.perf_takehome import|from candidates.CAND_$DEST_ID.perf_takehome import|" "$DEST_DIR/submission_tests.py"

# Ensure __init__.py exists
touch "$DEST_DIR/__init__.py"

echo "Copied $SOURCE_ID to $DEST_ID"
