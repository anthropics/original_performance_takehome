---
name: optimize
description: Run genetic algorithm optimization on the kernel
disable-model-invocation: true
---

# Genetic Algorithm Optimization

Optimize `build_kernel()` using genetic algorithm with mutate and crossover agents.

## Parameters (from $ARGUMENTS, format: --key=value)

- `--population=10` - number of candidates
- `--generations=10` - number of generations
- `--elite=3` - top candidates preserved unchanged
- `--crossover-rate=0.8` - probability of crossover
- `--mutation-rate=0.2` - probability of mutation

## Progress File

Log to `optimization_progress.txt` in root directory. This file enables resumption after interruption.

### State Detection on Start

1. **Read `optimization_progress.txt`** if it exists
2. **Check for existing candidates** via `ls candidates/`
3. **Determine resume point**:
   - No log file → fresh start
   - Log has `=== Generation N COMPLETE ===` → resume from generation N+1
   - Log without COMPLETE marker → re-evaluate and continue
   - Candidates exist but no log → evaluate existing, treat as generation 0 complete

## Helper Scripts

All scripts are in `./scripts/`:

### High-Level Scripts (use these)

| Script | Purpose | Output |
|--------|---------|--------|
| `init_population.sh {n}` | Initialize N candidates | "Created N candidates" |
| `plan_generation.sh {gen} {elite} {cr} {mr}` | Plan generation operations | ELITE/CROSSOVER/MUTATE/EVAL lines |
| `update_score.sh {id} {cycles}` | Update single score | Re-sorts scores.txt |
| `get_stats.sh [baseline]` | Get population stats | BEST/AVG/IMPROVEMENT lines |
| `eval_candidate.sh {id}` | Evaluate single candidate | Cycle count |
| `save_best.sh {id}` | Save best to `best/` | Copies perf_takehome.py |

### Low-Level Scripts (used by high-level scripts)

| Script | Purpose |
|--------|---------|
| `init_candidate.sh {id}` | Create single candidate folder |
| `eval_all.sh` | Evaluate all candidates |
| `get_elite.sh {n}` | Get top N candidate IDs |
| `get_non_elite.sh {n}` | Get non-elite IDs |
| `select_parents.sh {seed}` | Tournament selection |
| `should_crossover.sh {rate} {seed}` | Decide crossover |
| `should_mutate.sh {rate} {seed}` | Decide mutation |
| `parents_identical.sh {p1} {p2}` | Check if parents identical |

## Workflow

### 1. Resume Check

```bash
# Check for existing state
cat optimization_progress.txt 2>/dev/null
ls candidates/ 2>/dev/null
```

Log: `[RESUME] Starting from generation N` or `[START] Fresh optimization run`

### 2. Initialize (skip if candidates exist)

```bash
./scripts/init_population.sh 10
```

Log: `[INIT] Created 10 candidates`

### 3. Generation 0 - Baseline

```bash
# Evaluate one candidate (all are identical baseline)
BASELINE=$(./scripts/eval_candidate.sh 001)

# Set all scores to baseline
for id in 001 002 ... 010; do
  ./scripts/update_score.sh $id $BASELINE
done

# Save best
./scripts/save_best.sh 001
```

Log baseline and write `=== Generation 0 COMPLETE ===`

### 4. Per Generation Loop

For each generation 1 to N:

#### 4a. Get the Plan

```bash
./scripts/plan_generation.sh $GEN $ELITE $CROSSOVER_RATE $MUTATION_RATE
```

Example output:
```
ELITE: 001 002 003
CROSSOVER: 001 002 004
CROSSOVER: 001 003 005
MUTATE: 004
MUTATE: 006
EVAL: 004 005 006
```

Parse this output to get:
- `CROSSOVER` lines → crossover tasks
- `MUTATE` lines → mutation tasks
- `EVAL` line → candidates to re-evaluate

#### 4b. Execute Crossovers (PARALLEL)

**CRITICAL: Launch ALL crossover agents in ONE message**

For each `CROSSOVER: p1 p2 child` line:
```
Task(crossover, "CAND_{p1} CAND_{p2} CAND_{child}")
```

#### 4c. Execute Mutations (PARALLEL)

**CRITICAL: Launch ALL mutate agents in ONE message**

For each `MUTATE: id` line:
```
Task(mutate, "CAND_{id}")
```

#### 4d. Re-evaluate Modified Candidates (PARALLEL)

**CRITICAL: Run ALL evaluations in ONE message**

For each id in `EVAL` line:
```
Bash("./scripts/eval_candidate.sh {id}")
```

Then update scores:
```bash
./scripts/update_score.sh {id} {cycles}
```

#### 4e. Update Best & Log

```bash
./scripts/get_stats.sh $BASELINE
./scripts/save_best.sh $(./scripts/get_elite.sh 1)
```

Log generation summary and write `=== Generation N COMPLETE ===`

### 5. Finish

- Best kernel is in `best/perf_takehome.py`
- Report: initial cycles vs final cycles, improvement percentage

## Critical Rules

1. **PARALLEL CROSSOVERS**: Launch ALL crossover agents in ONE message
2. **PARALLEL MUTATIONS**: Launch ALL mutate agents in ONE message
3. **PARALLEL EVALS**: Run ALL evaluations in ONE message
4. **AGENT ARGS**: Pass ONLY candidate IDs - no hints, no bias
   - mutate: `CAND_{id}`
   - crossover: `CAND_{parent1} CAND_{parent2} CAND_{child}`

## Log Format

```
[START] Fresh optimization run | pop=10, gen=10, elite=3
[INIT] Created 10 candidates

=== Generation 0 ===
Baseline: 147734 cycles
=== Generation 0 COMPLETE ===

=== Generation 1 ===
[PLAN]
ELITE: 001 002 003
CROSSOVER: 001 002 004
MUTATE: 004 006
EVAL: 004 006

[CROSSOVER] 004: combined VLIW packing with memory layout
[MUTATE] 004: reordered memory accesses
[MUTATE] 006: unrolled inner loop
[EVAL] 004: 142000 cycles
[EVAL] 006: 148000 cycles

BEST: 004 142000
AVG: 145234
IMPROVEMENT: 3.9%
=== Generation 1 COMPLETE ===
```

**Key markers for resumption:**
- `=== Generation N COMPLETE ===` - generation fully finished
- `[EVAL] {id}:` - individual evaluations (mid-generation recovery)
