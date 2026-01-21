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

Check these in order:

1. **Read `optimization_progress.txt`** if it exists
2. **Check for existing candidates** via `ls candidates/`
3. **Determine resume point** based on log contents:
   - No log file → fresh start
   - Log exists with `=== Generation N COMPLETE ===` → resume from generation N+1
   - Log exists without COMPLETE marker → resume mid-generation (re-evaluate and continue)
   - Candidates exist but no log → evaluate existing candidates, treat as generation 0 complete

## Helper Scripts

- `./scripts/init_candidate.sh {id}` - initialize candidate folder
- `./scripts/should_crossover.sh {rate}` - exits 0 if should crossover, 1 if not
- `./scripts/should_mutate.sh {rate}` - exits 0 if should mutate, 1 if not

## Workflow

1. **Resume check**:
   - Read `optimization_progress.txt` if exists
   - Run `ls candidates/` to check for existing candidates
   - Determine resume point (see State Detection above)
   - Log: `[RESUME] Starting from generation N` or `[START] Fresh optimization run`

2. **Initialize** (skip if candidates already exist):
   - Run `./scripts/init_candidate.sh {id}` for candidates 001..N
   - Log: `[INIT] Created N candidates`

3. **Evaluate**:
   - Run `python candidates/{id}/submission_tests.py` to get cycle counts
   - Log each result: `[EVAL] {id}: XXXX cycles`

4. **Per generation**:
   - Rank candidates by cycles (lower = better)
   - Keep top `elite` unchanged
   - For each non-elite slot:
     - Use `./scripts/should_crossover.sh {rate}` to decide crossover vs copy
     - If crossover: select two parents from top half, create child
     - Use `./scripts/should_mutate.sh {rate}` to decide mutation
     - If mutate: apply mutation to candidate
   - Re-evaluate modified candidates
   - Write generation summary to log
   - **CRITICAL**: Write `=== Generation N COMPLETE ===` marker after generation finishes

5. **Finish**: Copy best `perf_takehome.py` to root, report improvement

## Critical Rules

- IMPORTANT: Launch ALL crossover agents in ONE message (parallel Task calls)
- IMPORTANT: Launch ALL mutate agents in ONE message (parallel Task calls)
- IMPORTANT: Pass ONLY candidate IDs to agents - no hints, no bias
  - mutate: `{id}`
  - crossover: `{parent1} {parent2} {child}`

## Log Format

```
[START] Fresh optimization run | pop=10, gen=10, elite=3
[INIT] Created 10 candidates

=== Generation 0 ===
Best: XXXX cycles | Avg: XXXX cycles

Candidates:
  001: XXXX cycles - baseline
  002: XXXX cycles - baseline
  ...
=== Generation 0 COMPLETE ===

=== Generation 1 ===
[CROSSOVER] 005 = 001 x 002: combined loop unrolling from 001 with register allocation from 002
[MUTATE] 006: reordered memory accesses
[EVAL] 005: XXXX cycles
[EVAL] 006: XXXX cycles

Best: XXXX cycles | Avg: XXXX cycles

Candidates:
  001: XXXX cycles - elite (unchanged)
  002: XXXX cycles - elite (unchanged)
  005: XXXX cycles - crossover of 001 x 002
  006: XXXX cycles - mutated
  ...
=== Generation 1 COMPLETE ===
```

**Key markers for resumption:**
- `=== Generation N COMPLETE ===` - signals generation fully finished (parse to find last complete)
- `[EVAL] {id}:` - individual evaluations (allows mid-generation recovery)

Record the one-line summary returned by each agent (mutation or crossover description).
