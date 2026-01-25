#!/bin/bash
# Reject the neighbor and clean up
# Usage: sa/scripts/reject_neighbor.sh
# Removes NEIGHBOR and increments rejected count

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
NEIGHBOR_DIR="$SA_DIR/candidates/CAND_NEIGHBOR"
STATE_FILE="$SA_DIR/candidates/state.txt"

if [ -d "$NEIGHBOR_DIR" ]; then
    rm -rf "$NEIGHBOR_DIR"
fi

# Increment rejected count
if [ -f "$STATE_FILE" ]; then
    REJECTED=$(grep "^REJECTED_COUNT=" "$STATE_FILE" | cut -d'=' -f2)
    NEW_REJECTED=$((REJECTED + 1))
    sed -i "s|^REJECTED_COUNT=.*|REJECTED_COUNT=$NEW_REJECTED|" "$STATE_FILE"
fi

echo "Rejected neighbor"
