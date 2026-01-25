---
name: optimize_sa
description: Run simulated annealing optimization on the kernel
disable-model-invocation: true
---

# Simulated Annealing Optimization

Optimize `build_kernel()` using simulated annealing with the mutate agent.

## Parameters

Pass parameters via `$ARGUMENTS` (format: `--key=value`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--initial-temp` | 5000 | Starting temperature |
| `--final-temp` | 10 | Stopping temperature |
| `--cooling-rate` | 0.95 | Multiplicative cooling (T' = T × rate) |
| `--iterations-per-temp` | 5 | Steps per temperature level |
| `--max-iterations` | 500 | Hard limit on total iterations |
| `--reset` | false | Clear existing state, start fresh |

## Workflow

### 1. Setup (once at start)

```bash
./sa/scripts/sa_setup.sh $ARGUMENTS
```

This configures parameters and optionally resets state.

### 2. Main Loop

```
loop:
    result = Bash("./sa/scripts/sa_step.sh")
    if "DONE" in result:
        break
    args = extract MUTATE_ARGS from result
    Task(mutate, args)
```

### Step-by-Step

1. **Run sa_step.sh**
   ```bash
   ./sa/scripts/sa_step.sh
   ```

2. **Check output**:
   - If output contains `DONE:` → optimization complete, stop
   - If output contains `MUTATE_ARGS:` → extract the args

3. **Call mutate agent** with the args (everything after "MUTATE_ARGS: "):
   ```
   Task(mutate, "sa CURRENT NEIGHBOR <step_category>")
   ```

4. **Repeat** from step 1

### Example Session

```
Bash: ./sa/scripts/sa_setup.sh --initial-temp=5000 --reset
Output:
  [CONFIG] Written to .../sa_config.sh
  [RESET] Ready for fresh start

Bash: ./sa/scripts/sa_step.sh
Output:
  [INIT] No state found, initializing...
  [STATUS] iter=0 temp=5000 current=147734 best=147734
  MUTATE_ARGS: sa CURRENT NEIGHBOR extensive

Task(mutate, "sa CURRENT NEIGHBOR extensive")
  → Agent creates NEIGHBOR with mutation

Bash: ./sa/scripts/sa_step.sh
Output:
  [ACCEPT] 25677 (current was 147734)
  [STATUS] iter=1 temp=5000 current=25677 best=25677
  MUTATE_ARGS: sa CURRENT NEIGHBOR extensive

... repeat ...

Bash: ./sa/scripts/sa_step.sh
Output:
  [STATUS] iter=500 temp=8.5 current=3827 best=3827
  DONE: Reached final temperature (best=3827)

→ Stop, report final result
```

## What sa_step.sh Does

Each call handles:
1. **Post-process previous iteration** (if NEIGHBOR exists):
   - Evaluate NEIGHBOR
   - Update best (if better than all-time best)
   - Accept/reject decision (Metropolis criterion)
   - Log iteration, update state
   - Cool temperature (every ITERATIONS_PER_TEMP iterations)

2. **Check termination**:
   - Temperature < FINAL_TEMP → DONE
   - Iteration >= MAX_ITERATIONS → DONE

3. **Prepare next mutation**:
   - Calculate step category from temperature
   - Output MUTATE_ARGS

## Critical Rules

1. **NEUTRAL PROMPTS**: Pass ONLY what sa_step.sh gives you to the mutate agent
   - NO scores, NO hints, NO "try to improve"
   - The step category is all the guidance needed

2. **SEQUENTIAL**: Wait for mutate agent to finish before calling sa_step.sh again

3. **TRUST THE SCRIPT**: sa_step.sh handles all SA logic - acceptance, cooling, logging
