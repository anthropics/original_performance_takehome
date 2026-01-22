---
name: optimize_sa
description: Run simulated annealing optimization on the kernel
disable-model-invocation: true
---

# Simulated Annealing Optimization

Optimize `build_kernel()` using simulated annealing with the mutate agent.

## Parameters (from $ARGUMENTS, format: --key=value)

- `--initial-temp=5000` - starting temperature (high enough for ~80% acceptance of bad moves)
- `--final-temp=10` - stopping temperature (low enough that only improvements accepted)
- `--cooling-rate=0.95` - multiplicative cooling factor (T' = T * rate), range 0.85-0.99
- `--iterations-per-temp=5` - perturbations per temperature level (Markov chain length)
- `--max-iterations=500` - hard limit on total iterations

### Parameter Tuning Guide

**Initial Temperature**: Should allow ~80% acceptance of "bad" moves initially.
- Formula: `T = typical_delta / -ln(0.8)` where typical_delta is expected cost difference
- For kernel optimization with deltas of ~1000 cycles: T ≈ 4500-5000

**Final Temperature**: Should be low enough that only improvements are accepted.
- Formula: `T = small_delta / -ln(0.01)` for 1% acceptance of small worsening
- For deltas of ~100 cycles: T ≈ 20-25, so T=10 is conservative

**Cooling Rate**: Controls exploration vs exploitation tradeoff.
- 0.99 = very slow cooling, more exploration, longer runtime
- 0.95 = balanced (recommended)
- 0.90 = faster cooling, may miss global optimum
- 0.85 = aggressive, risk of premature convergence

**Temperature Levels**: With defaults, `5000 * 0.95^n = 10` gives n ≈ 121 levels.
Total iterations: 121 × 5 = 605 (capped by max_iterations=500)

### Mutation Step Size

The mutation step size scales proportionally with temperature using textual categories:
- High temperature → larger mutations (exploration, escape local minima)
- Low temperature → smaller mutations (exploitation, fine-tuning)

Uses **logarithmic mapping** to match exponential cooling. With defaults (500 iterations):

| Category | Iterations | % of run | Max Scope |
|----------|------------|----------|-----------|
| extensive | ~105 | 21% | Major changes allowed |
| substantial | ~65 | 13% | Restructuring allowed |
| moderate | ~135 | 27% | Focused optimizations |
| small | ~135 | 27% | Local changes only |
| minimal | ~60 | 12% | Single tweaks only |

The category sets a ceiling - the agent may make smaller changes if appropriate.

This follows SA best practice: `ΔE_typical ≈ c · T` (typical energy change proportional to temperature)

## Algorithm Overview

Simulated annealing explores the solution space by:
1. Making small perturbations to the current solution
2. Always accepting improvements
3. Sometimes accepting worse solutions (controlled by temperature)
4. Gradually reducing temperature to focus on exploitation

Key difference from GA: SA maintains a single solution path rather than a population.

## Neutral Proposal Generation (CRITICAL)

**The mutate agent must receive NEUTRAL prompts with NO optimization bias.**

### Why This Matters

In SA, the **acceptance criterion** (Metropolis) handles exploration vs exploitation via temperature. The **proposal step** (mutation) should be neutral - it generates neighbors without preference for improvement.

If the orchestrator biases the mutate agent toward improvement:
- SA collapses into greedy hill-climbing
- Temperature becomes meaningless (exploration killed at the source)
- Algorithm gets trapped in local minima
- Theoretical guarantees are lost

### What to Pass to Mutate Agent

**ONLY these parameters:**
```
Task(mutate, "sa CAND_CURRENT CAND_NEIGHBOR $STEP_CATEGORY")
```

| Parameter | Source | Purpose |
|-----------|--------|---------|
| Base dir | "sa" | Directory context |
| Source | "CAND_CURRENT" | Which candidate to mutate |
| Dest | "CAND_NEIGHBOR" | Where to write result |
| Step category | calc_step_size.sh output | Mutation magnitude ceiling |

### What NOT to Pass

| Forbidden | Why |
|-----------|-----|
| Current score | Irrelevant to generating neighbors; induces greedy bias |
| Best score | Same as above |
| "Try to improve" | Converts SA to hill-climbing |
| "Push cycles down" | Greedy directive |
| Strategy suggestions | Narrows search space |
| Progress commentary | Anchoring bias |
| Emotional encouragement | Anthropomorphizes, adds implicit goals |

### Correct vs Incorrect Invocation

**WRONG** (biased):
```
Task(mutate, "sa CAND_CURRENT CAND_NEIGHBOR moderate - we're at 45000 cycles!
Try to push lower, maybe with loop unrolling or memory coalescing.")
```

**WRONG** (score leakage):
```
Task(mutate, "sa CAND_CURRENT CAND_NEIGHBOR extensive - current: 52000, best: 48000")
```

**CORRECT** (neutral):
```
Task(mutate, "sa CAND_CURRENT CAND_NEIGHBOR extensive")
```

The mutate agent will read the code, identify opportunities from the code itself, pick one randomly, and apply it. No external bias needed or wanted.

## Progress File

Log to `sa/optimization_progress.txt`. This file enables resumption after interruption.

### State Detection on Start

1. **Read `sa/candidates/state.txt`** if it exists
2. **Check for existing candidates** via `ls sa/candidates/`
3. **Determine resume point**:
   - No state file → fresh start
   - State file exists → resume from saved iteration/temperature

## Helper Scripts

All scripts are in `./sa/scripts/`:

### Core Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `check_state.sh` | Check if state exists for resumption | State values or "NO_STATE" |
| `init_solution.sh [temp] [--force]` | Initialize (skips if state exists) | "Initialized with X cycles" |
| `accept_solution.sh {cur} {new} {temp}` | Metropolis criterion | "ACCEPT" or "REJECT" |
| `accept_neighbor.sh {score}` | Move NEIGHBOR → CURRENT | Updates state |
| `reject_neighbor.sh` | Delete NEIGHBOR, increment reject count | "Rejected neighbor" |
| `update_best.sh {score}` | Update BEST if improved | "NEW_BEST" or "NO_CHANGE" |
| `update_temperature.sh {temp} {rate}` | Apply cooling | New temperature |
| `update_state.sh {key} {value}` | Update state.txt | "Updated key=value" |
| `get_stats.sh [baseline]` | Get current state | CURRENT/BEST/TEMP/ITERATION |
| `log_iteration.sh {iter} {temp} {cur} {best} {status}` | Log to history | "Logged iteration N" |
| `cleanup_neighbor.sh` | Remove NEIGHBOR directory | "Cleaned up" |
| `calc_step_size.sh {temp} {init} {final}` | Calculate mutation step category | Category string (minimal/small/moderate/substantial/extensive) |

### Wrapper Scripts (call shared utilities)

| Script | Purpose |
|--------|---------|
| `eval_candidate.sh {id}` | Evaluate candidate cycles |
| `copy_candidate.sh {src} {dest}` | Copy candidate folder |
| `init_candidate.sh {id}` | Create candidate folder |
| `save_best.sh {id}` | Save to `sa/best/` |

## Workflow

### 1. Resume Check

```bash
# Check for existing state
./sa/scripts/check_state.sh
```

**If state exists (exit code 0):**
- Read ITERATION and TEMPERATURE from state.txt
- Skip initialization
- Resume loop from saved iteration
- Log: `[RESUME] Continuing from iteration N at temperature T`

**If no state (exit code 1):**
- Proceed with fresh initialization
- Log: `[START] Fresh SA run`

### 2. Initialize (only if no state)

```bash
# This automatically skips if state exists
./sa/scripts/init_solution.sh $INITIAL_TEMP

# To force reset and start fresh:
./sa/scripts/init_solution.sh $INITIAL_TEMP --force
```

This creates:
- `sa/candidates/CAND_CURRENT/` - current solution
- `sa/candidates/CAND_BEST/` - best found so far
- `sa/candidates/state.txt` - current state
- `sa/candidates/history.txt` - iteration log

Log: `[INIT] Baseline: {cycles} cycles`

### 3. Main Optimization Loop

```
ITERATION = 0
while TEMPERATURE > FINAL_TEMP and ITERATION < MAX_ITERATIONS:
    for i in 1 to ITERATIONS_PER_TEMP:
        ITERATION += 1

        # 3a. Generate neighbor
        # 3b. Evaluate neighbor
        # 3c. Update best (BEFORE accept/reject!)
        # 3d. Accept/reject decision
        # 3e. Log iteration

    # 3f. Cool down
    TEMPERATURE = TEMPERATURE * COOLING_RATE
```

#### 3a. Generate Neighbor

Create a perturbation of CURRENT with step category proportional to temperature:

```bash
# Remove any existing neighbor first
./sa/scripts/cleanup_neighbor.sh

# Calculate step category based on current temperature (scales mutation magnitude)
STEP_CATEGORY=$(./sa/scripts/calc_step_size.sh $TEMPERATURE $INITIAL_TEMP $FINAL_TEMP)
```

Then launch the mutate agent with step category:
```
Task(mutate, "sa CAND_CURRENT CAND_NEIGHBOR $STEP_CATEGORY")
```

The mutate agent will:
1. Run `./scripts/copy_candidate.sh sa CURRENT NEIGHBOR`
2. Read CURRENT's code
3. Make a mutation scaled to STEP_CATEGORY (extensive at high temp, minimal at low temp)
4. Test correctness

#### 3b. Evaluate Neighbor

```bash
NEIGHBOR_SCORE=$(./sa/scripts/eval_candidate.sh NEIGHBOR)
```

#### 3c. Update Best (BEFORE accept/reject!)

**IMPORTANT**: Check if neighbor is better than all-time best BEFORE accept/reject.
We might reject a neighbor that's still our best solution ever!

```bash
# Save to BEST if this is the best we've ever seen (regardless of accept/reject)
./sa/scripts/update_best.sh $NEIGHBOR_SCORE
```

#### 3d. Accept/Reject Decision

```bash
# Get current score from state
CURRENT_SCORE=$(grep "^CURRENT_SCORE=" sa/candidates/state.txt | cut -d'=' -f2)
TEMPERATURE=$(grep "^TEMPERATURE=" sa/candidates/state.txt | cut -d'=' -f2)

# Metropolis criterion
DECISION=$(./sa/scripts/accept_solution.sh $CURRENT_SCORE $NEIGHBOR_SCORE $TEMPERATURE)

if [ "$DECISION" = "ACCEPT" ]; then
    ./sa/scripts/accept_neighbor.sh $NEIGHBOR_SCORE
else
    ./sa/scripts/reject_neighbor.sh
fi
```

#### 3e. Log Iteration

```bash
./sa/scripts/log_iteration.sh $ITERATION $TEMPERATURE $CURRENT_SCORE $BEST_SCORE $DECISION
./sa/scripts/update_state.sh ITERATION $ITERATION
```

#### 3f. Cool Down (after each temperature level)

```bash
NEW_TEMP=$(./sa/scripts/update_temperature.sh $TEMPERATURE $COOLING_RATE)
./sa/scripts/update_state.sh TEMPERATURE $NEW_TEMP
```

Log: `[COOL] Temperature: {old} → {new}`

### 4. Finish

```bash
./sa/scripts/save_best.sh BEST
./sa/scripts/get_stats.sh $BASELINE
```

- Best kernel is in `sa/best/perf_takehome.py`
- Report: initial cycles vs final cycles, improvement percentage, acceptance rate

## Critical Rules

1. **ONE MUTATION AT A TIME**: SA is sequential - process one neighbor per iteration
2. **CLEANUP BEFORE PERTURB**: Always run cleanup_neighbor.sh before creating a new neighbor
3. **NEUTRAL PROMPTS ONLY**: Pass ONLY the 4 specified parameters to mutate agent - NO scores, NO strategy hints, NO "try to improve". See "Neutral Proposal Generation" section.
4. **AGENT ARGS**: Pass base directory, candidate IDs, and step category
   - mutate: `sa CAND_CURRENT CAND_NEIGHBOR $STEP_CATEGORY`
   - Calculate STEP_CATEGORY using: `./sa/scripts/calc_step_size.sh $TEMPERATURE $INITIAL_TEMP $FINAL_TEMP`
5. **STATE PERSISTENCE**: Update state.txt after each iteration for resumption
6. **METROPOLIS CRITERION**: Use accept_solution.sh for all acceptance decisions
7. **STEP CATEGORY SCALING**: Always calculate step category before mutation - this ensures extensive exploration at high temp, minimal refinement at low temp

## State File Format

`sa/candidates/state.txt`:
```
TEMPERATURE=523.5
ITERATION=47
CURRENT_SCORE=12500
BEST_SCORE=11800
ACCEPTED_COUNT=32
REJECTED_COUNT=15
```

## History File Format

`sa/candidates/history.txt`:
```
# iteration temperature current_score best_score accepted
0 1000 147734 147734 INIT
1 1000 145000 145000 ACCEPT
2 1000 146000 145000 REJECT
3 1000 144500 144500 ACCEPT
...
```

## Log Format

```
[START] Fresh SA run | initial_temp=1000, final_temp=1, cooling_rate=0.95
[INIT] Baseline: 147734 cycles

=== Temperature 1000.0 (extensive) ===
[ITER 1] Perturbing CURRENT...
[MUTATE] extensive: optimized memory access pattern (major change)
[EVAL] NEIGHBOR: 145000 cycles
[ACCEPT] 145000 < 147734 (improvement)
[BEST] New best: 145000

[ITER 2] Perturbing CURRENT...
[MUTATE] extensive: unrolled inner loop
[EVAL] NEIGHBOR: 146500 cycles
[REJECT] 146500 > 145000, p=0.22 < rand=0.67

[ITER 3] Perturbing CURRENT...
[MUTATE] extensive: reduced register pressure
[EVAL] NEIGHBOR: 146000 cycles
[ACCEPT] 146000 > 145000, p=0.37 > rand=0.12 (Metropolis)

[COOL] Temperature: 1000.0 → 950.0

=== Temperature 100.0 (small) ===
[ITER 47] Perturbing CURRENT...
[MUTATE] small: tweaked loop bounds (local change)
...

=== Temperature 10.0 (minimal) ===
[ITER 120] Perturbing CURRENT...
[MUTATE] minimal: adjusted register allocation (single instruction tweak)
...

[FINISH]
Initial: 147734 cycles
Final best: 142000 cycles
Improvement: 3.9%
Acceptance rate: 67.8%
Total iterations: 150
```

## Comparison with GA

| Aspect | GA | SA |
|--------|-----|-----|
| Solutions | Population of 10 | Single solution |
| Operators | Crossover + Mutate | Mutate only |
| Selection | Pool-based, keep best N | Metropolis criterion |
| Parallelism | All offspring in parallel | Sequential |
| Exploration | Population diversity | Temperature-based |
| Control | Generations | Temperature schedule |
