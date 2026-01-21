---
name: optimize
description: Run genetic algorithm optimization on the kernel
disable-model-invocation: true
---

# Genetic Algorithm Optimization

Optimize `build_kernel()` using genetic algorithm with mutate and crossover agents.

## Parameters (from $ARGUMENTS, format: --key=value)

- `--population=10` - population size (kept constant each generation)
- `--generations=10` - number of generations
- `--offspring=7` - number of offspring per generation
- `--crossover-rate=0.8` - probability of crossover vs mutation
- `--mutation-rate=0.2` - (unused, crossover-rate determines split)

## Pool-Based Selection Model

This implementation uses a proper genetic algorithm with pool-based selection:

1. **Parents are never modified** - offspring are always new candidates
2. **Offspring get new IDs** - IDs grow over time (011, 012, etc.)
3. **Selection keeps best N** - after creating offspring, keep top `population` candidates
4. **Eliminated candidates are deleted** - cleanup after selection

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
| `plan_generation.sh {gen} {offspring} {cr} {mr}` | Plan offspring operations | CROSSOVER/MUTATE/EVAL lines |
| `update_score.sh {id} {cycles}` | Update single score | Re-sorts scores.txt |
| `get_stats.sh [baseline]` | Get population stats | BEST/AVG/IMPROVEMENT lines |
| `eval_candidate.sh {id}` | Evaluate single candidate | Cycle count |
| `save_best.sh {id}` | Save best to `best/` | Copies perf_takehome.py |
| `select_survivors.sh {n}` | Keep top N, delete rest | KEPT/DELETED lists |
| `copy_candidate.sh {src} {dest}` | Copy candidate folder | "Copied src to dest" |
| `next_candidate_id.sh` | Get next available ID | Next ID (e.g., "011") |

### Low-Level Scripts (used by high-level scripts)

| Script | Purpose |
|--------|---------|
| `init_candidate.sh {id}` | Create single candidate folder |
| `eval_all.sh` | Evaluate all candidates |
| `get_elite.sh {n}` | Get top N candidate IDs |
| `select_parents.sh {seed}` | Tournament selection |
| `should_crossover.sh {rate} {seed}` | Decide crossover vs mutation |
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
./scripts/plan_generation.sh $GEN $NUM_OFFSPRING $CROSSOVER_RATE $MUTATION_RATE
```

Example output:
```
CROSSOVER: 001 002 011
CROSSOVER: 001 003 012
MUTATE: 002 013
MUTATE: 001 014
EVAL: 011 012 013 014
```

Parse this output to get:
- `CROSSOVER` lines → crossover tasks (parent1, parent2, child_id)
- `MUTATE` lines → mutation tasks (parent, child_id)
- `EVAL` line → all new offspring to evaluate

**Note**: The plan creates placeholder directories for offspring, so next_candidate_id works correctly.

#### 4b. Execute Crossovers (PARALLEL)

**CRITICAL: Launch ALL crossover agents in ONE message**

For each `CROSSOVER: p1 p2 child` line:
```
Task(crossover, "CAND_{p1} CAND_{p2} CAND_{child}")
```

The crossover agent will:
1. Run `./scripts/copy_candidate.sh {p1} {child}` to copy parent1
2. Read both parents
3. Edit child to incorporate elements from parent2
4. Test correctness

#### 4c. Execute Mutations (PARALLEL)

**CRITICAL: Launch ALL mutate agents in ONE message**

For each `MUTATE: parent child` line:
```
Task(mutate, "CAND_{parent} CAND_{child}")
```

The mutate agent will:
1. Run `./scripts/copy_candidate.sh {parent} {child}` to copy parent
2. Read child file
3. Make ONE small mutation
4. Test correctness

#### 4d. Evaluate Offspring (PARALLEL)

**CRITICAL: Run ALL evaluations in ONE message**

For each id in `EVAL` line:
```
Bash("./scripts/eval_candidate.sh {id}")
```

Then update scores:
```bash
./scripts/update_score.sh {id} {cycles}
```

#### 4e. Select Survivors

```bash
./scripts/select_survivors.sh $POPULATION_SIZE
```

This keeps the top N candidates and **deletes** the rest, maintaining constant population size.

#### 4f. Update Best & Log

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
4. **NEVER MODIFY PARENTS**: Always create new candidates for offspring
5. **AGENT ARGS**: Pass ONLY candidate IDs - no hints, no bias
   - mutate: `CAND_{parent} CAND_{child}`
   - crossover: `CAND_{parent1} CAND_{parent2} CAND_{child}`

## Log Format

```
[START] Fresh optimization run | pop=10, gen=10, offspring=7
[INIT] Created 10 candidates

=== Generation 0 ===
Baseline: 147734 cycles
=== Generation 0 COMPLETE ===

=== Generation 1 ===
[PLAN]
CROSSOVER: 001 002 011
CROSSOVER: 001 003 012
MUTATE: 002 013
EVAL: 011 012 013

[CROSSOVER] 011: combined loop unrolling with VLIW packing
[CROSSOVER] 012: combined memory layout with instruction reordering
[MUTATE] 013: replaced modulo with bitwise AND
[EVAL] 011: 142000 cycles
[EVAL] 012: 145000 cycles
[EVAL] 013: 143500 cycles

[SELECT] Kept: 001 002 003 011 012 013 004 005 006 007
[SELECT] Deleted: 008 009 010

BEST: 011 142000
AVG: 144500
IMPROVEMENT: 3.9%
=== Generation 1 COMPLETE ===
```

**Key markers for resumption:**
- `=== Generation N COMPLETE ===` - generation fully finished
- `[EVAL] {id}:` - individual evaluations (mid-generation recovery)
- `[SELECT]` - selection completed
