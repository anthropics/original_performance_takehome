#!/bin/bash
# Plan all operations for a generation (pool-based selection)
# Usage: scripts/plan_generation.sh <generation> <num_offspring> <crossover_rate> <mutation_rate>
# Output:
#   CROSSOVER: parent1 parent2 child_id
#   MUTATE: parent child_id
#   EVAL: child_id1 child_id2 ...
#
# New approach: Creates NEW candidates for offspring, doesn't modify existing ones
# After evaluation, use select_survivors.sh to keep top N

if [ -z "$4" ]; then
    echo "Usage: $0 <generation> <num_offspring> <crossover_rate> <mutation_rate>"
    exit 1
fi

GEN="$1"
NUM_OFFSPRING="$2"
CROSSOVER_RATE="$3"
MUTATION_RATE="$4"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"
SCORES_FILE="$GA_DIR/candidates/scores.txt"

if [ ! -f "$SCORES_FILE" ]; then
    echo "ERROR: scores.txt not found. Run eval_all.sh first."
    exit 1
fi

# Track all new offspring IDs for evaluation
OFFSPRING_IDS=""

# Generate offspring
for i in $(seq 1 "$NUM_OFFSPRING"); do
    SEED="${GEN}_${i}"

    # Get next available ID for this offspring (using shared script)
    CHILD_ID=$("$PROJECT_ROOT/scripts/next_candidate_id.sh" ga)

    # Decide: crossover or mutation?
    if "$SCRIPT_DIR/should_crossover.sh" "$CROSSOVER_RATE" "$SEED"; then
        # Crossover: select two parents
        PARENTS=$("$SCRIPT_DIR/select_parents.sh" "$SEED")
        P1=$(echo "$PARENTS" | cut -d' ' -f1)
        P2=$(echo "$PARENTS" | cut -d' ' -f2)

        # Check if parents are identical - if so, do mutation instead
        if "$SCRIPT_DIR/parents_identical.sh" "$P1" "$P2"; then
            # Fall back to mutation
            echo "MUTATE: $P1 $CHILD_ID"
        else
            echo "CROSSOVER: $P1 $P2 $CHILD_ID"
        fi
    else
        # Mutation: select one parent
        PARENT=$("$SCRIPT_DIR/select_parents.sh" "$SEED" | cut -d' ' -f1)
        echo "MUTATE: $PARENT $CHILD_ID"
    fi

    OFFSPRING_IDS="$OFFSPRING_IDS $CHILD_ID"

    # Create placeholder directory so next_candidate_id increments properly
    mkdir -p "$GA_DIR/candidates/CAND_$CHILD_ID"
done

# Output which candidates need evaluation
OFFSPRING_IDS=$(echo "$OFFSPRING_IDS" | xargs)  # Trim whitespace
if [ -n "$OFFSPRING_IDS" ]; then
    echo "EVAL: $OFFSPRING_IDS"
fi
