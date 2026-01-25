---
name: optimize_lns
description: Run Large Neighborhood Search optimization on the kernel
disable-model-invocation: true
---

# Large Neighborhood Search Optimization

Optimize `build_kernel()` using LNS with kick_lns and refine_lns agents.

## Parameters

Pass parameters via `$ARGUMENTS` (format: `--key=value`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-iterations` | 100 | Hard limit on total iterations |
| `--reset` | false | Clear existing state, start fresh |

## Workflow

### 1. Setup (once at start)

```bash
./lns/scripts/lns_setup.sh $ARGUMENTS
```

This configures parameters and optionally resets state.

### 2. Main Loop

```
loop:
    result = Bash("./lns/scripts/lns_step.sh")
    if "DONE" in result:
        break
    if "KICK_ARGS" in result:
        args = extract KICK_ARGS from result
        kick_output = Task(kick_lns, args)
        result = Bash("./lns/scripts/lns_step.sh --agent-output=\"$kick_output\"")
    if "REFINE_ARGS" in result:
        args = extract REFINE_ARGS from result
        refine_output = Task(refine_lns, args)
        result = Bash("./lns/scripts/lns_step.sh --agent-output=\"$refine_output\"")
```

### Step-by-Step

1. **Run lns_step.sh**
   ```bash
   ./lns/scripts/lns_step.sh
   ```

2. **Check output**:
   - If output contains `DONE:` → optimization complete, stop
   - If output contains `KICK_ARGS:` → extract args, call kick_lns
   - If output contains `REFINE_ARGS:` → extract args, call refine_lns

3. **Call appropriate agent** with the args (everything after "KICK_ARGS: " or "REFINE_ARGS: "):
   ```
   Task(kick_lns, "lns CURRENT NEIGHBOR <operator>")
   ```
   or
   ```
   Task(refine_lns, "lns NEIGHBOR NEIGHBOR")
   ```

4. **Feed agent output back to lns_step.sh**:
   ```bash
   ./lns/scripts/lns_step.sh --agent-output="<agent output>"
   ```

5. **Repeat** from step 2

### Example Session

```
Bash: ./lns/scripts/lns_setup.sh --max-iterations=50 --reset
Output:
  [CONFIG] Written to .../lns_config.sh
  [RESET] Ready for fresh start

Bash: ./lns/scripts/lns_step.sh
Output:
  [INIT] No state found, initializing...
  [STATUS] iter=0 phase=kick current=147734 best=147734
  KICK_ARGS: lns CURRENT NEIGHBOR novel | Generate orthogonal optimization approach

Task(kick_lns, "lns CURRENT NEIGHBOR novel | Generate orthogonal optimization approach")
  → Agent creates NEIGHBOR with large mutation
  → Returns: "NOVEL: loop_tiling | Break loops into cache-friendly tiles\nDONE: Applied 4x4 loop tiling"

Bash: ./lns/scripts/lns_step.sh --agent-output="NOVEL: loop_tiling | ..."
Output:
  [OPERATOR] Added: loop_tiling | Break loops into cache-friendly tiles
  [KICK] Complete, transitioning to refine
  [STATUS] iter=0 phase=refine current=147734 best=147734
  REFINE_ARGS: lns NEIGHBOR NEIGHBOR

Task(refine_lns, "lns NEIGHBOR NEIGHBOR")
  → Agent refines kicked solution
  → Returns: "DONE: Micro-optimized (final: 25000 cycles)"

Bash: ./lns/scripts/lns_step.sh --agent-output="DONE: ..."
Output:
  [ACCEPT] New current: 25000 cycles
  [BEST] New best: 25000 cycles
  [STATUS] iter=1 phase=kick current=25000 best=25000
  KICK_ARGS: lns CURRENT NEIGHBOR loop_tiling | Break loops into cache-friendly tiles

... repeat ...

Bash: ./lns/scripts/lns_step.sh
Output:
  [STATUS] iter=50 phase=kick current=3827 best=3827
  DONE: Reached max iterations (best=3827)

→ Stop, report final result
```

## What lns_step.sh Does

Each call handles:
1. **Process previous agent output** (if provided):
   - For kick: parse NOVEL output, add new operator with dedup, transition to refine
   - For refine: evaluate result, accept as current, update best if improved, increment iteration

2. **Check termination**:
   - Iteration >= MAX_ITERATIONS → DONE

3. **Output next action**:
   - kick phase → select random operator, output KICK_ARGS
   - refine phase → output REFINE_ARGS

## Critical Rules

1. **NEUTRAL PROMPTS**: Pass ONLY what lns_step.sh gives you to the agents
   - NO scores, NO hints, NO "try to improve"
   - The operator is all the guidance kick_lns needs
   - refine_lns gets no guidance (blind refinement)

2. **SEQUENTIAL**: Wait for each agent to finish before calling lns_step.sh again

3. **TRUST THE SCRIPT**: lns_step.sh handles all LNS logic - operator selection, state, termination
