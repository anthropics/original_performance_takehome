#!/bin/bash
# Decide whether to accept a neighbor solution using Metropolis criterion
# Usage: sa/scripts/accept_solution.sh <current_score> <neighbor_score> <temperature>
# Output: "ACCEPT" or "REJECT"
#
# Metropolis criterion:
# - If neighbor is better (lower score): always accept
# - If neighbor is worse: accept with probability exp(-delta/T)

if [ -z "$3" ]; then
    echo "Usage: $0 <current_score> <neighbor_score> <temperature>"
    exit 1
fi

CURRENT_SCORE="$1"
NEIGHBOR_SCORE="$2"
TEMPERATURE="$3"

# Use Python for reliable floating point math
python3 << EOF
import random
import math

current = $CURRENT_SCORE
neighbor = $NEIGHBOR_SCORE
temp = $TEMPERATURE

delta = neighbor - current

if delta <= 0:
    # Improvement - always accept
    print("ACCEPT")
elif temp <= 0:
    # Temperature is zero - only accept improvements
    print("REJECT")
else:
    # Worse solution - accept with probability exp(-delta/temp)
    probability = math.exp(-delta / temp)
    if random.random() < probability:
        print("ACCEPT")
    else:
        print("REJECT")
EOF
