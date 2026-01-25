#!/bin/bash
# Select two parents for crossover using tournament selection
# Usage: scripts/select_parents.sh <seed>
# Output: "{parent1} {parent2}" (IDs)
# Requires: candidates/scores.txt must exist (run eval_all.sh first)
#
# Selection method: Tournament selection from top half of population
# - Randomly pick 2 candidates from top half, select better one as parent1
# - Repeat for parent2 (ensuring different from parent1)

if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SCORES_FILE="$ROOT_DIR/candidates/scores.txt"

if [ ! -f "$SCORES_FILE" ]; then
    echo "ERROR: scores.txt not found. Run eval_all.sh first."
    exit 1
fi

# Get top half of candidates (selection pool)
TOTAL=$(wc -l < "$SCORES_FILE")
POOL_SIZE=$(( (TOTAL + 1) / 2 ))  # Round up

# Use Python for deterministic random selection
python3 << EOF
import random

random.seed("$SEED")

# Read scores (already sorted by cycles ascending)
scores = []
with open("$SCORES_FILE") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            scores.append((parts[0], int(parts[1])))

# Sort by cycles (ascending = better)
scores.sort(key=lambda x: x[1])

# Top half is our selection pool
pool_size = $POOL_SIZE
pool = scores[:pool_size]

# Tournament selection for parent1
idx1, idx2 = random.sample(range(len(pool)), 2)
parent1 = pool[idx1] if pool[idx1][1] <= pool[idx2][1] else pool[idx2]

# Tournament selection for parent2 (must be different)
while True:
    idx1, idx2 = random.sample(range(len(pool)), 2)
    parent2 = pool[idx1] if pool[idx1][1] <= pool[idx2][1] else pool[idx2]
    if parent2[0] != parent1[0]:
        break

print(f"{parent1[0]} {parent2[0]}")
EOF
