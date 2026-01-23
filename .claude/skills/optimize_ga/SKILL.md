---
name: optimize_ga
description: Run genetic algorithm optimization on the kernel
disable-model-invocation: true
---

# Genetic Algorithm Optimization

Optimize `build_kernel()` using genetic algorithm with mutate and crossover agents.

## Parameters

Pass parameters via `$ARGUMENTS` (format: `--key=value`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--population` | 10 | Population size (kept constant) |
| `--generations` | 10 | Number of generations |
| `--offspring` | 7 | Offspring per generation |
| `--crossover-rate` | 0.8 | Probability of crossover vs mutation |
| `--mutation-rate` | 0.2 | Mutation probability |
| `--reset` | false | Clear existing state, start fresh |

## Workflow

### 1. Setup (once at start)

```bash
./ga/scripts/ga_setup.sh $ARGUMENTS
```

This configures parameters and optionally resets state.

### 2. Main Loop

```
loop:
    result = Bash("./ga/scripts/ga_step.sh")
    if "DONE" in result:
        break
    tasks = extract all TASK lines from result
    launch ALL tasks in parallel:
        for each "TASK: crossover ..." -> Task(crossover, args)
        for each "TASK: mutate ..." -> Task(mutate, args)
```

### Step-by-Step

1. **Run ga_step.sh**
   ```bash
   ./ga/scripts/ga_step.sh
   ```

2. **Check output**:
   - If output contains `DONE:` → optimization complete, stop
   - If output contains `TASK:` lines → extract all tasks

3. **Launch ALL agents in parallel** (CRITICAL - one message with all Task calls):
   - For `TASK: crossover ga CAND_P1 CAND_P2 CAND_CHILD`:
     ```
     Task(crossover, "ga CAND_P1 CAND_P2 CAND_CHILD")
     ```
   - For `TASK: mutate ga CAND_PARENT CAND_CHILD`:
     ```
     Task(mutate, "ga CAND_PARENT CAND_CHILD")
     ```

4. **Wait for all agents to complete**, then **repeat** from step 1

### Example Session

```
Bash: ./ga/scripts/ga_setup.sh --population=10 --generations=5 --reset
Output:
  [CONFIG] Written to .../ga_config.sh
  [RESET] Ready for fresh start

Bash: ./ga/scripts/ga_step.sh
Output:
  [INIT] No state found, initializing population...
  [INIT] Baseline: 147734 cycles
  [STATUS] gen=0 best=147734 avg=147734
  TASK: crossover ga CAND_001 CAND_002 CAND_011
  TASK: crossover ga CAND_001 CAND_003 CAND_012
  TASK: mutate ga CAND_002 CAND_013
  TASK: mutate ga CAND_001 CAND_014

Launch in ONE message:
  Task(crossover, "ga CAND_001 CAND_002 CAND_011")
  Task(crossover, "ga CAND_001 CAND_003 CAND_012")
  Task(mutate, "ga CAND_002 CAND_013")
  Task(mutate, "ga CAND_001 CAND_014")
  → All agents run in parallel

Bash: ./ga/scripts/ga_step.sh
Output:
  [EVAL] 011: 142000 cycles
  [EVAL] 012: 145000 cycles
  [EVAL] 013: 143500 cycles
  [EVAL] 014: 140000 cycles
  [SELECT] Kept: 014 011 013 012 001 002 003 004 005 006
  [STATUS] gen=1 best=140000 avg=143500
  TASK: crossover ga CAND_014 CAND_011 CAND_015
  TASK: mutate ga CAND_014 CAND_016
  ...

... repeat ...

Bash: ./ga/scripts/ga_step.sh
Output:
  [STATUS] gen=5 best=3827 avg=4500
  DONE: Completed 5 generations (best=3827, improvement=97.4%)

→ Stop, report final result
```

## What ga_step.sh Does

Each call handles:
1. **Initialize** (first call only):
   - Create population of N candidates
   - Evaluate baseline (all identical)
   - Set up state tracking

2. **Post-process previous generation** (if offspring exist):
   - Evaluate all offspring
   - Update scores
   - Select survivors (keep top N)
   - Save best candidate
   - Log generation complete

3. **Check termination**:
   - Generation >= MAX_GENERATIONS → DONE

4. **Plan next generation**:
   - Decide crossover vs mutation for each offspring
   - Select parents (tournament selection)
   - Output all TASK lines

## Critical Rules

1. **PARALLEL AGENTS**: Launch ALL Task calls from one ga_step.sh output in a SINGLE message

2. **NEUTRAL PROMPTS**: Pass ONLY what ga_step.sh gives you to agents
   - NO scores, NO hints, NO "try to improve"

3. **WAIT FOR COMPLETION**: All agents must finish before calling ga_step.sh again

4. **TRUST THE SCRIPT**: ga_step.sh handles all GA logic - selection, evaluation, survivor management

## Best Solution

After completion, find the best kernel at: `ga/best/perf_takehome.py`
