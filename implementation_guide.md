# Implementation Guide

## Rules of Engagement
- Do not modify anything under `tests/`, including `tests/frozen_problem.py` and `tests/submission_tests.py`.
- Performance is measured against the frozen simulator; treat `tests/frozen_problem.py` as canonical.
- Only optimize the kernel generator: `perf_takehome.py` `KernelBuilder.build_kernel`.
- Keep correctness aligned with `reference_kernel2` in `problem.py`.
- Use `python tests/submission_tests.py` as the only validation gate; record cycle count.
- Avoid "multicore fixes" — `N_CORES = 1` is intentional in `problem.py`.
- Debug tooling (`watch_trace.py`, pause/debug slots) can be used locally but must not affect submission logic.

## Architecture Constraints

| Resource | Limit per Cycle |
|----------|-----------------|
| VALU slots | 6 |
| ALU slots | 12 |
| Load slots | 2 |
| Store slots | 2 |
| Flow slots | 1 |
| VLEN | 8 |
| Scratch space | 1536 words |

## Operations Cost Table

| Operation | Slots Used | Cycles | Notes |
|-----------|------------|--------|-------|
| **VALU ops** | 1 VALU/op | 1 | Can pack up to 6 per cycle |
| vbroadcast | 1 VALU | 1 | Broadcast scalar to vector |
| valu (+, -, *, ^, &, %, <, ==, etc) | 1 VALU | 1 | Element-wise vector ops |
| multiply_add | 1 VALU | 1 | `dest = a * b + c` fused |
| **Load ops** | 1 Load/op | 1 | Max 2 per cycle |
| load (scalar) | 1 Load | 1 | mem[scratch[addr]] -> scratch[dest] |
| load_offset (gather element) | 1 Load | 1 | mem[scratch[addr+offset]] -> scratch[dest+offset] |
| vload (contiguous) | 1 Load | 1 | Load 8 contiguous words |
| const | 1 Load | 1 | Load immediate to scratch |
| **Store ops** | 1 Store/op | 1 | Max 2 per cycle |
| store (scalar) | 1 Store | 1 | scratch[src] -> mem[scratch[addr]] |
| vstore (contiguous) | 1 Store | 1 | Store 8 contiguous words |
| **Flow ops** | 1 Flow | 1 | Max 1 per cycle (BOTTLENECK) |
| select (scalar) | 1 Flow | 1 | Conditional select |
| vselect (vector) | 1 Flow | 1 | Vector conditional select |
| add_imm | 1 Flow | 1 | `dest = a + immediate` |
| **ALU ops** | 1 ALU/op | 1 | Can pack up to 12 per cycle |
| alu (+, -, *, etc) | 1 ALU | 1 | Scalar arithmetic |

## Operation Optimizations (Algebraic Rewrites)

### 1. Replace vselect with Arithmetic

**Problem**: `vselect` uses the Flow slot (1/cycle max) - major bottleneck.

**Pattern 1: Branch direction (val % 2 == 0 ? 1 : 2)**
```
BEFORE (3 cycles, uses Flow):
  tmp1 = val % 2        # VALU
  tmp1 = (tmp1 == 0)    # VALU  
  offset = vselect(tmp1, 1, 2)  # FLOW ← bottleneck

AFTER (2 cycles, VALU only):
  tmp1 = val & 1        # VALU: 0 if even, 1 if odd
  offset = 1 + tmp1     # VALU: 1 if even, 2 if odd
```
**Savings**: 1 cycle per batch, eliminates Flow dependency

**Pattern 2: Wrap to zero (idx >= n_nodes ? 0 : idx)**
```
BEFORE (2 cycles, uses Flow):
  tmp1 = idx < n_nodes  # VALU: 1 if valid, 0 if wrap
  idx = vselect(tmp1, idx, 0)  # FLOW ← bottleneck

AFTER (2 cycles, VALU only):
  tmp1 = idx < n_nodes  # VALU: 1 if valid, 0 if wrap
  idx = idx * tmp1      # VALU: idx if valid, 0 if wrap
```
**Savings**: Eliminates Flow dependency, enables better packing

### 2. Use multiply_add for Fused Operations

**Problem**: `idx = 2*idx + offset` takes 2 cycles.

```
BEFORE (2 cycles):
  idx = idx * 2         # VALU
  idx = idx + offset    # VALU (data dependency)

AFTER (1 cycle):
  idx = multiply_add(idx, 2, offset)  # VALU: idx*2 + offset
```
**Savings**: 1 cycle per batch

### 3. Use Bitwise AND Instead of Modulo

**Problem**: `val % 2` for checking even/odd.

```
BEFORE:
  tmp = val % 2   # VALU (modulo operation)

AFTER:  
  tmp = val & 1   # VALU (bitwise AND - simpler)
```
**Savings**: Same cycle count, but simpler operation

### 4. Vector Copy with VALU Instead of Scalar ALU

**Problem**: Copying vectors element-by-element with scalar ALU.

```
BEFORE (8 cycles):
  for i in range(8):
    s_idx[off+i] = v_idx[i]  # 4 scalar ALU ops per cycle

AFTER (1 cycle):
  s_idx[off:] = v_idx + v_zero  # VALU: vector add with zero
```
**Savings**: 7 cycles per copy operation

## Key Bottlenecks
1. **Flow slot = 1/cycle**: vselect serializes. Avoid when possible with arithmetic.
2. **Load slot = 2/cycle**: Gathers (load_offset) are expensive. 8 gathers per vector = 4 cycles.
3. **Memory latency**: Loads complete at end of cycle (VLIW semantics).
4. **Data dependencies**: Can't merge ops where output feeds input.

## Progress Log

### Session 1 (Previous)
- Implemented vectorized kernel: 6592 cycles

### Session 2 (Current)
| Change | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Baseline | 147,734 | 1× | Original scalar code |
| Basic SIMD + 2-batch | 10,656 | 13.8× | VLEN=8 vectorization |
| Software pipelining | 8,496 | 17.4× | Prefetch during hash |
| VALU vector copy | 6,704 | 22× | Replace scalar ALU loop |
| Broadcast opt (r0,r11) | 6,660 | 22.2× | Single load for broadcast |
| 3-batch processing | 6,078 | 24.3× | Use 6 VALU slots |
| vselect→arithmetic | 5,566 | 26.5× | Replace branch vselect |
| multiply_add fusion | 5,334 | 27.7× | Fuse idx*2+offset |
| wrap vselect→multiply | 5,054 | 29.2× | Replace wrap vselect |
| 3-batch broadcast | 4,294 | 34.4× | Optimize broadcast rounds |

## Target Thresholds
| Target | Cycles | Speedup |
|--------|--------|---------|
| Baseline | 147,734 | 1× |
| **Current** | **4,294** | **34.4×** |
| Opus 4 many hours | <2,164 | 68× |
| Opus 4.5 casual | <1,790 | 83× |
| Sonnet 4.5 many hours | <1,548 | 95× |
| Opus 4.5 2hr | <1,579 | 94× |
| Opus 4.5 11hr | <1,487 | 99× |
| Opus 4.5 improved | <1,363 | 108× |

## Current Performance
- Cycle count: 4,294
- Speedup: 34.4×
- Status: Correctness OK; need ~2× more improvement for Opus 4

## Remaining Optimizations

### Approaches Explored (Not Viable)

**Speculative Child Prefetch**: Load both children during hash, select correct one.
- Issue: Need 48 loads (3 batches × 8 elements × 2 children) but only 24 slots available in hash
- Would require reducing batch size (2 instead of 3), losing more cycles than saved

**Index Deduplication**: Use broadcast for rounds with few unique indices.
- Issue: Unique indices depend on hash values (data-dependent), not tree structure
- Only rounds 0 and 11 are guaranteed all-zero (already optimized as broadcast)

### Remaining Opportunities

1. **Cross-round pipelining**: After round N finishes, start loading round N+1's nodes
   - Challenge: Indices aren't known until index update completes

2. **Idle load slot utilization**: 4393 wasted load slots per kernel
   - Post-hash index update has 6 VALU-only cycles with idle loads

3. **Better remainder handling**: 2 vectors processed sequentially (12 cycles)
   - Could batch both vectors together (6 cycles)

4. **Bundle merging**: Many single-VALU bundles could be merged

### Bottleneck Analysis

The fundamental limit is the gather operations (random memory access):
- 14 gather rounds × ~170 cycles = 2380 cycles in gathers alone
- Hash computation is fully overlapped with prefetch
- VALU is not the bottleneck (6 slots, plenty of capacity)

## External References
- Designing a SIMD Algorithm from Scratch: https://mcyoung.xyz/2023/11/27/simd-base64/
- AVX-512 Multi-hash Computation (Intel): https://networkbuilders.intel.com/docs/networkbuilders/intel-avx-512-ultra-parallelized-multi-hash-computation-for-data-streaming-workloads-technology-guide-1693301077.pdf
- Software Pipelining (Lam 1988): https://dl.acm.org/doi/10.1145/960116.54022
- Iron Law of Performance: https://blog.codingconfessions.com/p/one-law-to-rule-all-code-optimizations

## Known Notes
- Pre-existing lint warning: unused import `defaultdict` in `perf_takehome.py`.
