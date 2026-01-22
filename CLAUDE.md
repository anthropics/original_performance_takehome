# CLAUDE.md - Performance Optimization Project

## Project Overview

This is Anthropic's original performance take-home challenge. The goal is to **minimize clock cycles** in a simulated VLIW SIMD machine by optimizing the kernel implementation in `perf_takehome.py`.

## Critical Constraints

**NEVER MODIFY THE `tests/` FOLDER.** The submission tests use a frozen copy of the simulator (`tests/frozen_problem.py`) to prevent cheating. Any modifications to `tests/` will invalidate your solution.

### Validation Commands
```bash
# Verify tests/ folder is unchanged (should be empty output)
git diff origin/main tests/

# Run official submission tests
python tests/submission_tests.py
```

## Project Structure

```
.
├── perf_takehome.py      # YOUR OPTIMIZATION TARGET - KernelBuilder.build_kernel()
├── problem.py            # Simulator and reference implementations (can read, study)
├── tests/
│   ├── submission_tests.py   # Official verification (DO NOT MODIFY)
│   └── frozen_problem.py     # Frozen simulator copy (DO NOT MODIFY)
├── watch_trace.py        # Trace visualization server
└── watch_trace.html      # Perfetto UI
```

## Performance Benchmarks (Lower is Better)

| Cycles | Achievement |
|--------|-------------|
| 147,734 | Baseline (starting point) |
| 18,532 | Previous 2-hour starting point |
| 2,164 | Claude Opus 4 after many hours |
| 1,790 | Claude Opus 4.5 casual session |
| 1,579 | Claude Opus 4.5 after 2 hours |
| 1,548 | Claude Sonnet 4.5 after many hours |
| 1,487 | Claude Opus 4.5 after 11.5 hours |
| 1,363 | Claude Opus 4.5 improved harness |
| ??? | Best human (better than all above) |

## Machine Architecture

### VLIW SIMD Architecture
- **VLIW**: Multiple "engines" execute in parallel per cycle
- **SIMD**: Vector instructions operate on VLEN=8 elements
- **Single-core only**: N_CORES=1 (multicore is intentionally disabled)

### Slot Limits Per Cycle
```python
SLOT_LIMITS = {
    "alu": 12,      # 12 scalar ALU operations
    "valu": 6,      # 6 vector ALU operations
    "load": 2,      # 2 load operations
    "store": 2,     # 2 store operations
    "flow": 1,      # 1 flow control operation
    "debug": 64,    # Debug (ignored in submission)
}
```

### Available Instructions

**ALU (Scalar)**: `+`, `-`, `*`, `//`, `%`, `^`, `&`, `|`, `<<`, `>>`, `<`, `==`

**VALU (Vector)**:
- `vbroadcast` - Broadcast scalar to vector
- `multiply_add` - Fused multiply-add
- All ALU ops work on vectors

**Load**:
- `load` - Scalar load from memory
- `vload` - Vector load (8 contiguous elements)
- `const` - Load immediate constant
- `load_offset` - Load with offset

**Store**:
- `store` - Scalar store to memory
- `vstore` - Vector store (8 contiguous elements)

**Flow**:
- `select` / `vselect` - Conditional select
- `add_imm` - Add immediate
- `jump`, `cond_jump`, `cond_jump_rel` - Control flow
- `halt`, `pause` - Execution control

### Key Constants
```python
VLEN = 8           # Vector length
SCRATCH_SIZE = 1536  # Register/cache space
```

## The Algorithm

The kernel performs parallel tree traversal with hashing:

```
for each round (16 rounds):
    for each batch item (256 items):
        1. Load idx and val from memory
        2. Load node_val from forest[idx]
        3. val = hash(val XOR node_val)  # 6-stage hash
        4. idx = 2*idx + (1 if val%2==0 else 2)  # Navigate tree
        5. idx = 0 if idx >= n_nodes else idx  # Wrap if overflow
        6. Store val and idx back to memory
```

### Hash Function (6 stages)
Each stage: `val = op2(op1(val, const1), op3(val, const2))`

## Optimization Strategies

### 1. Vectorization (VLEN=8)
- Process 8 batch items in parallel using vector instructions
- Use `vload`/`vstore` for contiguous memory access
- Use `vbroadcast` to convert scalars to vectors

### 2. Instruction Packing (VLIW)
- Pack multiple independent operations into single instruction bundles
- Current baseline uses 1 slot per instruction (huge waste)
- Can use up to 12 ALU + 6 VALU + 2 load + 2 store + 1 flow per cycle

### 3. Loop Optimization
- Consider loop unrolling
- Reduce loop overhead with `cond_jump_rel`
- Pre-compute constants outside loops

### 4. Memory Access Optimization
- Minimize loads/stores (2 slots per cycle is the bottleneck)
- Use scratch space efficiently as cache
- Batch memory operations

### 5. Algorithm-level
- Reduce instruction dependencies
- Pipeline operations across iterations
- Consider mathematical simplifications

## Development Workflow

### Quick Test Cycle
```bash
# Run performance test (shows cycles)
python perf_takehome.py Tests.test_kernel_cycles

# Run with trace output
python perf_takehome.py Tests.test_kernel_trace
```

### Full Verification
```bash
# Official submission tests (correctness + all performance tiers)
python tests/submission_tests.py
```

### Trace Visualization
1. Generate trace: `python perf_takehome.py Tests.test_kernel_trace`
2. Start viewer: `python watch_trace.py`
3. Open browser, click "Open Perfetto"
4. Re-run test to hot-reload trace

## Files You Can Modify

- `perf_takehome.py` - Main optimization target
- `problem.py` - Only for adding debug helpers (submission uses frozen copy)

## Files You CANNOT Modify

- `tests/submission_tests.py` - Official verification
- `tests/frozen_problem.py` - Frozen simulator

## Common Pitfalls

1. **Modifying tests/**: Invalidates your solution
2. **Enabling multicore**: N_CORES=1 is intentional
3. **Not verifying correctness**: Always run submission_tests.py
4. **Forgetting slot limits**: Can't exceed SLOT_LIMITS per cycle
5. **Data dependencies**: Effects don't apply until cycle end

## Quick Reference

```python
# Slot limits reminder
alu: 12, valu: 6, load: 2, store: 2, flow: 1

# Vector length
VLEN = 8

# Memory layout
mem[0] = rounds
mem[1] = n_nodes
mem[2] = batch_size
mem[3] = forest_height
mem[4] = forest_values_p  # pointer to tree values
mem[5] = inp_indices_p    # pointer to indices
mem[6] = inp_values_p     # pointer to values
```
