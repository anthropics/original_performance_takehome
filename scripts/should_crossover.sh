#!/bin/bash
# Deterministically decide if crossover should occur
# Usage: scripts/should_crossover.sh <probability> <seed>
# Returns: exit code 0 if yes, 1 if no

PROB="${1:-0.8}"
SEED="${2:-$RANDOM}"

RESULT=$(python3 -c "import random; random.seed('$SEED'); print(1 if random.random() < $PROB else 0)")
exit $((1 - RESULT))
