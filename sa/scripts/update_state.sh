#!/bin/bash
# Update the SA state file
# Usage: sa/scripts/update_state.sh <key> <value>
# Updates a single key in state.txt

if [ -z "$2" ]; then
    echo "Usage: $0 <key> <value>"
    exit 1
fi

KEY="$1"
VALUE="$2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
STATE_FILE="$SA_DIR/candidates/state.txt"

if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: state.txt not found"
    exit 1
fi

# Update the key (or add if not present)
if grep -q "^$KEY=" "$STATE_FILE"; then
    sed -i "s|^$KEY=.*|$KEY=$VALUE|" "$STATE_FILE"
else
    echo "$KEY=$VALUE" >> "$STATE_FILE"
fi

echo "Updated $KEY=$VALUE"
