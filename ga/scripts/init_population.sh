#!/bin/bash
# Initialize a population of N candidates
# Usage: ga/scripts/init_population.sh <n>
# Output: "Created N candidates"

if [ -z "$1" ]; then
    echo "Usage: $0 <population_size>"
    exit 1
fi

N="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"

# Create candidates directory with __init__.py
mkdir -p "$GA_DIR/candidates"
touch "$GA_DIR/candidates/__init__.py"

# Initialize each candidate using shared script
for i in $(seq 1 "$N"); do
    # Use 10# prefix to force decimal interpretation (avoids octal issues with 08, 09)
    ID=$(printf "%03d" "$((10#$i))")
    "$PROJECT_ROOT/scripts/init_candidate.sh" ga "$ID"
    touch "$GA_DIR/candidates/CAND_$ID/__init__.py"
done

echo "Created $N candidates"
