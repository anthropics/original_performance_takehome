# Optimization Logs for VLIW Kernel Performance

## Instructions for Logging

After every optimization attempt, add a new entry **at the top** of the log section (below this instructions block) with the following format:

```
## Attempt [N]
**Cycles:** [cycle count]
**Speedup:** [X]x over baseline (147,734)

### Optimizations Tried
- ✅ [Successful optimization description]
- ❌ [Failed optimization description and why it failed]

### Notes
[Any additional observations or insights]

---
```

Where [N] is 1 greater than the previous attempt number.

---

# Log Entries

## Attempt 12
**Cycles:** 11,124
**Speedup:** 13.3x over baseline (147,734)

### Optimizations Tried
- ✅ Overlap A's vselect (flow) operations with B's early hash stages 0-1 (saves 4 cycles per iteration)
- ✅ Pack A's store with B's hash stage 2 (already had hash stage 0, moved to stage 2)
- ❌ Deep software pipelining (load next-A during B hash stages 3-5) - caused out-of-bounds memory access on last iteration

### Notes
This is the final working version. Achieved ~13.3x speedup. Further pipelining across iteration boundaries proved too complex and caused correctness issues due to loading beyond batch boundaries.

---

## Attempt 11
**Cycles:** 12,148
**Speedup:** 12.2x over baseline

### Optimizations Tried
- ✅ Overlap A's store with B's hash stage 0 (pack store engine with valu engine)

### Notes
Simple optimization that saved ~250 cycles by utilizing idle store engine during B's first hash cycle.

---

## Attempt 10
**Cycles:** 12,404
**Speedup:** 11.9x over baseline

### Optimizations Tried
- ✅ Aggressive interleaving of A hash stages 0-4 with B setup (load, address compute, address copy, tree loads)
- ✅ Pack A hash stage 2 with B address copy (alu + valu in same cycle)
- ✅ Pack A hash stages 3-4 with B tree loads (load + valu in same cycle)

### Notes
Major restructure to overlap A's hash computation with B's entire setup phase. This is the key insight that reduced cycles significantly.

---

## Attempt 9
**Cycles:** 14,196
**Speedup:** 10.4x over baseline

### Optimizations Tried
- ✅ Overlap Store A with Load B (different engines: store vs load)

### Notes
Simple win - store and load engines can work in parallel.

---

## Attempt 8
**Cycles:** 14,452
**Speedup:** 10.2x over baseline

### Optimizations Tried
- ✅ 2x inner loop unrolling - process batches A and B per iteration
- ✅ Separate tree address registers for A and B

### Notes
2x unrolling reduces loop overhead by half. Requires double the vector registers but SCRATCH_SIZE (1536) is sufficient.

---

## Attempt 7
**Cycles:** 14,948
**Speedup:** 9.9x over baseline

### Optimizations Tried
- ✅ Pre-broadcast forest_values_p (v_fvp) outside the inner loop
- ✅ Remove per-iteration vbroadcast for forest_values_p

### Notes
Hoisting loop-invariant broadcasts outside the loop saves 1 cycle per inner iteration.

---

## Attempt 6
**Cycles:** 15,460
**Speedup:** 9.5x over baseline

### Optimizations Tried
- ✅ Pack outer loop init: load (const) + alu (base pointer setup) in 1 cycle

### Notes
Small optimization combining reset operations at start of each round.

---

## Attempt 5
**Cycles:** 15,492
**Speedup:** 9.5x over baseline

### Optimizations Tried
- ✅ Pack hash broadcasts: 6 vbroadcast ops per cycle (was 2 per cycle)
- ✅ Use add_imm for outer loop counter increment (no improvement, same cycle count as alu)
- ❌ Pack ALU comparison with FLOW cond_jump - FAILED (read-after-write hazard: cond_jump reads tmp1 before alu writes it)

### Notes
Discovered that packing alu+flow with data dependency doesn't work because "Effects of instructions don't take effect until the end of cycle."

---

## Attempt 4
**Cycles:** 15,496
**Speedup:** 9.5x over baseline

### Optimizations Tried
- ✅ Pack (val & 1) with (idx * 2) in same valu cycle (independent operations)

### Notes
These operations write to different registers and have no dependencies.

---

## Attempt 3
**Cycles:** 16,008
**Speedup:** 9.2x over baseline

### Optimizations Tried
- ✅ Pack vstore with alu counter updates (store + alu engines in same cycle)

### Notes
Store engine and ALU engine can run in parallel.

---

## Attempt 2
**Cycles:** 17,544
**Speedup:** 8.4x over baseline

### Optimizations Tried
- ✅ Pack 8 address copy operations into 1 ALU cycle (alu has 12 slots)

### Notes
ALU engine has high slot capacity (12), so 8 operations easily fit.

---

## Attempt 1
**Cycles:** 21,128
**Speedup:** 7.0x over baseline

### Optimizations Tried
- ✅ SIMD vectorization: process 8 elements per iteration using vload/vstore/valu
- ✅ Vector constants (v_zero, v_one, v_two, v_n_nodes) via vbroadcast
- ✅ Pack 2 vloads in same load cycle
- ✅ Pack 2 vstores in same store cycle  
- ✅ Pack 2 scalar loads (tree values) per cycle
- ✅ Hash optimization: pack (op1, op3) in first valu cycle, op2 in second
- ✅ Loop structure with cond_jump instead of full unrolling
- ✅ Hash constant vectors pre-broadcast before loop

### Notes
First working vectorized implementation. Major jump from scalar baseline. Key insight: VLEN=8 allows processing 8 batch elements in parallel. Tree value loads remain scattered (8 loads at 2/cycle = 4 cycles).

---

## Baseline
**Cycles:** 147,734
**Speedup:** 1.0x (reference)

### Implementation
Original scalar implementation with full loop unrolling. Each batch element processed individually with no SIMD or instruction packing.

---

# Architecture Reference

| Engine | Slots/cycle | Notes |
|--------|-------------|-------|
| alu    | 12          | Scalar operations |
| valu   | 6           | Vector ops (VLEN=8) |
| load   | 2           | Includes vload, const |
| store  | 2           | Includes vstore |
| flow   | 1           | Jumps, vselect |

**Key Constraints:**
- Effects don't take place until end of cycle (no read-after-write in same cycle)
- N_CORES = 1 (no multi-core parallelism available)
- SCRATCH_SIZE = 1536

**Test Parameters:** forest_height=10, rounds=16, batch_size=256

**Test Thresholds:**
- ✅ < 147,734 (baseline) - PASSED
- ✅ < 18,532 (updated starting point) - PASSED  
- ❌ < 2,164 (opus4_many_hours)
- ❌ < 1,790 (opus45_casual)
- ❌ < 1,579 (opus45_2hr)
- ❌ < 1,548 (sonnet45_many_hours)
- ❌ < 1,487 (opus45_11hr)
- ❌ < 1,363 (opus45_improved_harness)
