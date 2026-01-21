#!/bin/bash
# Initialize a population of N candidates
# Usage: scripts/init_population.sh <n>
# Output: "Created N candidates"

if [ -z "$1" ]; then
    echo "Usage: $0 <population_size>"
    exit 1
fi

N="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Create candidates directory with __init__.py
mkdir -p "$ROOT_DIR/candidates"
touch "$ROOT_DIR/candidates/__init__.py"

# Initialize each candidate
for i in $(seq 1 "$N"); do
    # Use 10# prefix to force decimal interpretation (avoids octal issues with 08, 09)
    ID=$(printf "%03d" "$((10#$i))")
    "$SCRIPT_DIR/init_candidate.sh" "$ID"
    touch "$ROOT_DIR/candidates/CAND_$ID/__init__.py"
done

echo "Created $N candidates"
