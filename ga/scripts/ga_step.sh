#!/bin/bash
# GA Step - Single step handler for Genetic Algorithm
# Usage: ./ga/scripts/ga_step.sh
#
# This script handles EVERYTHING except the actual mutations/crossovers:
# 1. Initialize population if needed
# 2. Post-process previous generation (evaluate, select survivors)
# 3. Check termination conditions
# 4. Plan next generation and output all TASK lines
#
# Output format:
#   [STATUS] gen=N best=B avg=A
#   TASK: crossover ga CAND_P1 CAND_P2 CAND_CHILD
#   TASK: mutate ga CAND_PARENT CAND_CHILD <step_size>
# Or:
#   [STATUS] ...
#   DONE: <reason> (best=B)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"

# Source config
source "$SCRIPT_DIR/ga_config.sh"

STATE_FILE="$GA_DIR/candidates/state.txt"
SCORES_FILE="$GA_DIR/candidates/scores.txt"
PROGRESS_FILE="$GA_DIR/optimization_progress.txt"

# Helper: read state variable
get_state() {
    grep "^$1=" "$STATE_FILE" 2>/dev/null | cut -d'=' -f2
}

# Helper: update state variable
set_state() {
    if grep -q "^$1=" "$STATE_FILE" 2>/dev/null; then
        sed -i "s|^$1=.*|$1=$2|" "$STATE_FILE"
    else
        echo "$1=$2" >> "$STATE_FILE"
    fi
}

# Helper: log to progress file
log_progress() {
    echo "$1" >> "$PROGRESS_FILE"
}

# Helper: check if offspring file is populated (not just placeholder)
is_offspring_ready() {
    local ID="$1"
    local KERNEL_FILE="$GA_DIR/candidates/CAND_$ID/perf_takehome.py"
    if [ -f "$KERNEL_FILE" ] && [ -s "$KERNEL_FILE" ]; then
        return 0
    fi
    return 1
}

# ============================================================
# PHASE 1: Initialize if needed
# ============================================================
if [ ! -f "$STATE_FILE" ]; then
    echo "[INIT] No state found, initializing population with diverse mutations..."

    # Create candidates directory
    mkdir -p "$GA_DIR/candidates"
    touch "$GA_DIR/candidates/__init__.py"

    # Initialize population (creates baseline copies)
    "$SCRIPT_DIR/init_population.sh" "$POPULATION" > /dev/null

    # Evaluate baseline BEFORE mutations
    BASELINE=$("$PROJECT_ROOT/scripts/eval_candidate.sh" ga 001 2>/dev/null)
    echo "[INIT] Baseline: $BASELINE cycles"

    # Initialize state - set phase to init_mutate to trigger diversity mutations
    cat > "$STATE_FILE" << EOF
GENERATION=0
BASELINE=$BASELINE
PHASE=init_mutate
OFFSPRING_IDS=
EOF

    # Log to progress file
    log_progress "[START] GA optimization | pop=$POPULATION, gen=$GENERATIONS, offspring=$OFFSPRING"
    log_progress "[INIT] Baseline: $BASELINE cycles"
    log_progress "[INIT] Creating diverse initial population..."

    # Output mutation tasks for all candidates (except 001 which stays as baseline)
    # Use "extensive" step category for wild/diverse initial mutations
    for i in $(seq 2 "$POPULATION"); do
        ID=$(printf "%03d" "$((10#$i))")
        echo "TASK: mutate ga CAND_001 CAND_$ID extensive"
        log_progress "[INIT_MUTATE] 001 -> $ID (extensive)"
    done

    exit 0
fi

# ============================================================
# PHASE 1b: Post-process initial mutations
# ============================================================
PHASE=$(get_state PHASE)
if [ "$PHASE" = "init_mutate" ]; then
    BASELINE=$(get_state BASELINE)
    echo "[INIT] Evaluating initial diverse population..."

    # Evaluate all candidates (001 is baseline, others are mutated)
    for i in $(seq 1 "$POPULATION"); do
        ID=$(printf "%03d" "$((10#$i))")
        KERNEL_FILE="$GA_DIR/candidates/CAND_$ID/perf_takehome.py"

        if [ -f "$KERNEL_FILE" ] && [ -s "$KERNEL_FILE" ]; then
            CYCLES=$("$PROJECT_ROOT/scripts/eval_candidate.sh" ga "$ID" 2>/dev/null || echo "ERROR")
            if [ "$CYCLES" != "ERROR" ]; then
                "$SCRIPT_DIR/update_score.sh" "$ID" "$CYCLES" > /dev/null
                log_progress "[EVAL] $ID: $CYCLES cycles"
            else
                # If mutation failed, reset to baseline
                echo "[WARN] Failed to evaluate $ID, resetting to baseline"
                cp "$GA_DIR/candidates/CAND_001/perf_takehome.py" "$GA_DIR/candidates/CAND_$ID/"
                "$SCRIPT_DIR/update_score.sh" "$ID" "$BASELINE" > /dev/null
            fi
        else
            # If file missing, copy baseline
            cp "$GA_DIR/candidates/CAND_001/perf_takehome.py" "$GA_DIR/candidates/CAND_$ID/"
            "$SCRIPT_DIR/update_score.sh" "$ID" "$BASELINE" > /dev/null
        fi
    done

    # Get stats and save best
    STATS=$("$SCRIPT_DIR/get_stats.sh" "$BASELINE")
    BEST_ID=$(echo "$STATS" | grep "^BEST:" | awk '{print $2}')
    BEST_CYCLES=$(echo "$STATS" | grep "^BEST:" | awk '{print $3}')

    "$SCRIPT_DIR/save_best.sh" "$BEST_ID" > /dev/null

    log_progress ""
    log_progress "=== Generation 0 (Initial Diversity) ==="
    log_progress "BEST: $BEST_ID $BEST_CYCLES"
    log_progress "=== Generation 0 COMPLETE ==="
    log_progress ""

    # Transition to normal breeding phase
    set_state PHASE "breed"
fi

# ============================================================
# PHASE 2: Post-process previous generation (if offspring exist)
# ============================================================
PHASE=$(get_state PHASE)
GENERATION=$(get_state GENERATION)
BASELINE=$(get_state BASELINE)
OFFSPRING_IDS=$(get_state OFFSPRING_IDS)

if [ "$PHASE" = "postbreed" ] && [ -n "$OFFSPRING_IDS" ]; then
    log_progress "=== Generation $((GENERATION + 1)) ==="

    # Evaluate all offspring
    for ID in $OFFSPRING_IDS; do
        if is_offspring_ready "$ID"; then
            CYCLES=$("$PROJECT_ROOT/scripts/eval_candidate.sh" ga "$ID" 2>/dev/null || echo "ERROR")
            if [ "$CYCLES" != "ERROR" ]; then
                "$SCRIPT_DIR/update_score.sh" "$ID" "$CYCLES" > /dev/null
                log_progress "[EVAL] $ID: $CYCLES cycles"
            else
                echo "[WARN] Failed to evaluate $ID, removing from population"
                rm -rf "$GA_DIR/candidates/CAND_$ID"
            fi
        else
            echo "[WARN] Offspring $ID not ready (agent may have failed), removing"
            rm -rf "$GA_DIR/candidates/CAND_$ID"
        fi
    done

    # Select survivors
    SELECTION=$("$SCRIPT_DIR/select_survivors.sh" "$POPULATION")
    KEPT=$(echo "$SELECTION" | grep "^KEPT:" | cut -d: -f2)
    DELETED=$(echo "$SELECTION" | grep "^DELETED:" | cut -d: -f2)
    log_progress "[SELECT] Kept:$KEPT"
    if [ -n "$DELETED" ] && [ "$DELETED" != " (none - population already at or below target size)" ]; then
        log_progress "[SELECT] Deleted:$DELETED"
    fi

    # Get stats and save best
    STATS=$("$SCRIPT_DIR/get_stats.sh" "$BASELINE")
    BEST_ID=$(echo "$STATS" | grep "^BEST:" | awk '{print $2}')
    BEST_CYCLES=$(echo "$STATS" | grep "^BEST:" | awk '{print $3}')
    AVG=$(echo "$STATS" | grep "^AVG:" | awk '{print $2}')
    IMPROVEMENT=$(echo "$STATS" | grep "^IMPROVEMENT:" | awk '{print $2}')

    "$SCRIPT_DIR/save_best.sh" "$BEST_ID" > /dev/null

    log_progress ""
    log_progress "BEST: $BEST_ID $BEST_CYCLES"
    log_progress "AVG: $AVG"
    log_progress "IMPROVEMENT: $IMPROVEMENT"

    # Increment generation
    NEW_GEN=$((GENERATION + 1))
    set_state GENERATION "$NEW_GEN"
    set_state PHASE "breed"
    set_state OFFSPRING_IDS ""
    GENERATION=$NEW_GEN

    log_progress "=== Generation $GENERATION COMPLETE ==="
    log_progress ""
fi

# ============================================================
# PHASE 3: Check termination conditions
# ============================================================
GENERATION=$(get_state GENERATION)
BASELINE=$(get_state BASELINE)

# Get current stats
if [ -f "$SCORES_FILE" ]; then
    STATS=$("$SCRIPT_DIR/get_stats.sh" "$BASELINE" 2>/dev/null || echo "")
    BEST_ID=$(echo "$STATS" | grep "^BEST:" | awk '{print $2}')
    BEST_CYCLES=$(echo "$STATS" | grep "^BEST:" | awk '{print $3}')
    AVG=$(echo "$STATS" | grep "^AVG:" | awk '{print $2}')
    echo "[STATUS] gen=$GENERATION best=$BEST_CYCLES avg=$AVG"
else
    echo "[STATUS] gen=$GENERATION (no scores yet)"
fi

# Check if done
if [ "$GENERATION" -ge "$GENERATIONS" ]; then
    IMPROVEMENT=$("$SCRIPT_DIR/get_stats.sh" "$BASELINE" | grep "^IMPROVEMENT:" | awk '{print $2}')
    echo "DONE: Completed $GENERATIONS generations (best=$BEST_CYCLES, improvement=$IMPROVEMENT)"
    set_state PHASE "done"
    exit 0
fi

# ============================================================
# PHASE 4: Plan next generation and output TASKs
# ============================================================

# Plan the generation
NEXT_GEN=$((GENERATION + 1))
PLAN=$("$SCRIPT_DIR/plan_generation.sh" "$NEXT_GEN" "$OFFSPRING" "$CROSSOVER_RATE" "$MUTATION_RATE")

# Log plan
log_progress "[PLAN] Generation $NEXT_GEN"

# Collect offspring IDs for later evaluation
EVAL_LINE=$(echo "$PLAN" | grep "^EVAL:" || echo "")
OFFSPRING_IDS=$(echo "$EVAL_LINE" | cut -d: -f2 | xargs)

# Update state for post-processing
set_state PHASE "postbreed"
set_state OFFSPRING_IDS "$OFFSPRING_IDS"

# Output crossover tasks (using process substitution to avoid subshell)
while read -r line; do
    P1=$(echo "$line" | awk '{print $2}')
    P2=$(echo "$line" | awk '{print $3}')
    CHILD=$(echo "$line" | awk '{print $4}')
    echo "TASK: crossover ga CAND_$P1 CAND_$P2 CAND_$CHILD"
    log_progress "[CROSSOVER] $P1 + $P2 -> $CHILD"
done < <(echo "$PLAN" | grep "^CROSSOVER:" || true)

# Output mutation tasks (using process substitution to avoid subshell)
while read -r line; do
    [ -z "$line" ] && continue
    PARENT=$(echo "$line" | awk '{print $2}')
    CHILD=$(echo "$line" | awk '{print $3}')
    STEP_SIZE=$(echo "$line" | awk '{print $4}')
    echo "TASK: mutate ga CAND_$PARENT CAND_$CHILD $STEP_SIZE"
    log_progress "[MUTATE] $PARENT -> $CHILD ($STEP_SIZE)"
done < <(echo "$PLAN" | grep "^MUTATE:" || true)
