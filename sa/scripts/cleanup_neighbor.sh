#!/bin/bash
# Clean up the neighbor candidate after rejection
# Usage: sa/scripts/cleanup_neighbor.sh
# Removes CAND_NEIGHBOR directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
NEIGHBOR_DIR="$SA_DIR/candidates/CAND_NEIGHBOR"

if [ -d "$NEIGHBOR_DIR" ]; then
    rm -rf "$NEIGHBOR_DIR"
    echo "Cleaned up NEIGHBOR"
else
    echo "No NEIGHBOR to clean up"
fi
