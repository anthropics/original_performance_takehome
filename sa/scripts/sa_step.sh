#!/bin/bash
# SA Step - Single iteration handler for Simulated Annealing
# Usage: ./sa/scripts/sa_step.sh
#
# This script handles EVERYTHING except the mutation itself:
# 1. Post-process previous iteration (if NEIGHBOR exists)
# 2. Check termination conditions
# 3. Output MUTATE_ARGS for next iteration (or DONE)
#
# Output format:
#   [STATUS] iter=N temp=T current=C best=B
#   MUTATE_ARGS: sa CURRENT NEIGHBOR <step_category>
# Or:
#   [STATUS] ...
#   DONE: <reason> (best=B)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SA_DIR")"

# Source config
source "$SCRIPT_DIR/sa_config.sh"

STATE_FILE="$SA_DIR/candidates/state.txt"
NEIGHBOR_DIR="$SA_DIR/candidates/CAND_NEIGHBOR"

# Helper: read state variable
get_state() {
    grep "^$1=" "$STATE_FILE" | cut -d'=' -f2
}

# Helper: update state variable
set_state() {
    sed -i "s|^$1=.*|$1=$2|" "$STATE_FILE"
}

# ============================================================
# PHASE 1: Initialize if needed
# ============================================================
if [ ! -f "$STATE_FILE" ]; then
    echo "[INIT] No state found, initializing..."
    "$SCRIPT_DIR/init_solution.sh" "$INITIAL_TEMP"
fi

# ============================================================
# PHASE 2: Post-process previous iteration (if NEIGHBOR exists)
# ============================================================
if [ -d "$NEIGHBOR_DIR" ]; then
    ITERATION=$(get_state ITERATION)
    TEMPERATURE=$(get_state TEMPERATURE)
    CURRENT_SCORE=$(get_state CURRENT_SCORE)
    BEST_SCORE=$(get_state BEST_SCORE)

    # Evaluate neighbor
    NEIGHBOR_SCORE=$("$PROJECT_ROOT/scripts/eval_candidate.sh" sa NEIGHBOR 2>/dev/null || echo "ERROR")

    if [ "$NEIGHBOR_SCORE" = "ERROR" ]; then
        echo "[EVAL] NEIGHBOR evaluation failed, treating as rejection"
        "$SCRIPT_DIR/reject_neighbor.sh" > /dev/null
        DECISION="REJECTED"
        NEIGHBOR_SCORE=999999
    else
        # Update best BEFORE accept/reject
        BEST_RESULT=$("$SCRIPT_DIR/update_best.sh" "$NEIGHBOR_SCORE")
        if [[ "$BEST_RESULT" == NEW_BEST* ]]; then
            BEST_SCORE="$NEIGHBOR_SCORE"
            echo "[BEST] New best: $NEIGHBOR_SCORE cycles"
        fi

        # Accept/reject decision
        DECISION=$("$SCRIPT_DIR/accept_solution.sh" "$CURRENT_SCORE" "$NEIGHBOR_SCORE" "$TEMPERATURE")

        if [ "$DECISION" = "ACCEPT" ]; then
            "$SCRIPT_DIR/accept_neighbor.sh" "$NEIGHBOR_SCORE" > /dev/null
            CURRENT_SCORE="$NEIGHBOR_SCORE"
            echo "[ACCEPT] $NEIGHBOR_SCORE (current was $CURRENT_SCORE)"
        else
            "$SCRIPT_DIR/reject_neighbor.sh" > /dev/null
            echo "[REJECT] $NEIGHBOR_SCORE (keeping current $CURRENT_SCORE)"
        fi
    fi

    # Increment iteration
    NEW_ITERATION=$((ITERATION + 1))
    set_state ITERATION "$NEW_ITERATION"

    # Log iteration
    "$SCRIPT_DIR/log_iteration.sh" "$NEW_ITERATION" "$TEMPERATURE" "$CURRENT_SCORE" "$BEST_SCORE" "$DECISION" > /dev/null

    # Cool temperature every ITERATIONS_PER_TEMP iterations
    if [ $((NEW_ITERATION % ITERATIONS_PER_TEMP)) -eq 0 ]; then
        NEW_TEMP=$("$SCRIPT_DIR/update_temperature.sh" "$TEMPERATURE" "$COOLING_RATE")
        set_state TEMPERATURE "$NEW_TEMP"
        echo "[COOL] Temperature: $TEMPERATURE -> $NEW_TEMP"
    fi
fi

# ============================================================
# PHASE 3: Check termination conditions
# ============================================================
ITERATION=$(get_state ITERATION)
TEMPERATURE=$(get_state TEMPERATURE)
CURRENT_SCORE=$(get_state CURRENT_SCORE)
BEST_SCORE=$(get_state BEST_SCORE)

echo "[STATUS] iter=$ITERATION temp=$TEMPERATURE current=$CURRENT_SCORE best=$BEST_SCORE"

# Check if done
DONE_REASON=""
if python3 -c "exit(0 if $TEMPERATURE < $FINAL_TEMP else 1)" 2>/dev/null; then
    DONE_REASON="Reached final temperature"
elif [ "$ITERATION" -ge "$MAX_ITERATIONS" ]; then
    DONE_REASON="Reached max iterations"
fi

if [ -n "$DONE_REASON" ]; then
    echo "DONE: $DONE_REASON (best=$BEST_SCORE)"
    exit 0
fi

# ============================================================
# PHASE 4: Prepare next mutation
# ============================================================

# Cleanup any leftover neighbor (shouldn't exist but be safe)
if [ -d "$NEIGHBOR_DIR" ]; then
    rm -rf "$NEIGHBOR_DIR"
fi

# Calculate step size based on temperature
STEP_CATEGORY=$("$SCRIPT_DIR/calc_step_size.sh" "$TEMPERATURE" "$INITIAL_TEMP" "$FINAL_TEMP")

echo "MUTATE_ARGS: sa CURRENT NEIGHBOR $STEP_CATEGORY"
