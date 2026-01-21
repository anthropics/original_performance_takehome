#!/bin/bash
# Plan all operations for a generation
# Usage: scripts/plan_generation.sh <generation> <elite_count> <crossover_rate> <mutation_rate>
# Output:
#   ELITE: 001 002 003
#   CROSSOVER: parent1 parent2 child
#   MUTATE: candidate
# Requires: candidates/scores.txt must exist

if [ -z "$4" ]; then
    echo "Usage: $0 <generation> <elite_count> <crossover_rate> <mutation_rate>"
    exit 1
fi

GEN="$1"
ELITE_COUNT="$2"
CROSSOVER_RATE="$3"
MUTATION_RATE="$4"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SCORES_FILE="$ROOT_DIR/candidates/scores.txt"

if [ ! -f "$SCORES_FILE" ]; then
    echo "ERROR: scores.txt not found. Run eval_all.sh first."
    exit 1
fi

# Get elite and non-elite candidates
ELITE=$("$SCRIPT_DIR/get_elite.sh" "$ELITE_COUNT" | tr '\n' ' ' | sed 's/ $//')
NON_ELITE=$("$SCRIPT_DIR/get_non_elite.sh" "$ELITE_COUNT")

echo "ELITE: $ELITE"

# Track which candidates will be modified (for mutation decisions)
MODIFIED=""

# For each non-elite slot, decide crossover
for SLOT in $NON_ELITE; do
    SEED="${GEN}_${SLOT}"

    if "$SCRIPT_DIR/should_crossover.sh" "$CROSSOVER_RATE" "$SEED"; then
        PARENTS=$("$SCRIPT_DIR/select_parents.sh" "$SEED")
        P1=$(echo "$PARENTS" | cut -d' ' -f1)
        P2=$(echo "$PARENTS" | cut -d' ' -f2)

        # Skip if parents are identical
        if ! "$SCRIPT_DIR/parents_identical.sh" "$P1" "$P2"; then
            echo "CROSSOVER: $P1 $P2 $SLOT"
            MODIFIED="$MODIFIED $SLOT"
        fi
    fi
done

# For each non-elite slot, decide mutation
for SLOT in $NON_ELITE; do
    MUTATE_SEED="${GEN}_${SLOT}_m"

    if "$SCRIPT_DIR/should_mutate.sh" "$MUTATION_RATE" "$MUTATE_SEED"; then
        echo "MUTATE: $SLOT"
        # Add to modified if not already there
        if [[ ! " $MODIFIED " =~ " $SLOT " ]]; then
            MODIFIED="$MODIFIED $SLOT"
        fi
    fi
done

# Output which candidates need re-evaluation
MODIFIED=$(echo "$MODIFIED" | xargs)  # Trim whitespace
if [ -n "$MODIFIED" ]; then
    echo "EVAL: $MODIFIED"
fi
