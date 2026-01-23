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

### Session 2
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

### Session 3
| Change | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Start | 4,294 | 34.4× | From previous session |
| Init batching | 4,272 | 34.6× | Batch const loads, hash broadcasts |
| Vector broadcasts | 4,150 | 35.6× | 5 broadcasts in one bundle |
| XOR+addr merge | 4,010 | 36.8× | Merge XOR with next-triple addr |
| Broadcast remainder | 4,094→3,996 | 37.0× | Batch remainder ops |
| Remainder XOR | 3,996 | 37.0× | Batch XOR operations |
| Triple-0 loads | 4,030 | 36.7× | Batch v_node[2] loads |

### Session 4
| Change | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Start | 4,030 | 36.7× | From commit 9b66c24 |
| Remainder index batching | 3,946 | 37.4× | Batch all remainder index ops |

### Session 5
| Change | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Hash fusion + select rounds | 3,765 | 39.2× | Fuse stages 0,2,4; round 1/12 select | 
| BATCH=6 for all rounds | 3,241 | 45.6× | Fill VALU slots across rounds |
| Wider prefetch window | 2,905 | 50.9× | Pack gathers into index-update bundles |

### Session 6 (Final - Zolotukhin Algorithm)
| Change | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Automatic list scheduler | 1,307 | 113.0× | Greedy VLIW bundle packing |
| vselect for levels 0-3 | 1,307 | 113.0× | Preload nodes 0-14, use flow ops |
| Init phase merge | 1,305 | 113.2× | Let scheduler interleave init with main |

**Final: 1,305 cycles (113.2× speedup)**

## Final Solution Architecture

The final solution uses a fundamentally different approach: **automatic list scheduling**.

### Key Innovations

1. **Flat Operation List + Greedy Scheduler**
   - Generate all operations as `(engine, slot)` tuples
   - Greedy algorithm packs into VLIW bundles respecting dependencies
   - Handles RAW, WAW, WAR hazards automatically

2. **vselect for Tree Levels 0-3**
   - Preload tree nodes 0-14 as broadcast vectors during init
   - Use `vselect` (flow ops) instead of memory gathers for levels 0-3
   - Trades 15 broadcasts for eliminating ~1200 gather operations

3. **Tiled Processing**
   - `group_size=17`: Process 17 blocks concurrently
   - `round_tile=13`: Process 13 rounds per tile
   - Optimal values found empirically

4. **Hash Stage Fusion**
   - Fusable pattern: `op1=='+' and op2=='+' and op3=='<<'`
   - `multiply_add(val, (1 + (1 << shift)), const)` replaces 3 ops with 1
   - Stages 0, 2, 4 are fusable; stages 1, 3, 5 require 3 ops each

### Performance Breakdown

| Metric | Value |
|--------|-------|
| Total VALU ops | 7,267 |
| Theoretical minimum | 1,212 cycles |
| Achieved | 1,305 cycles |
| Efficiency | 92.9% |
| VALU utilization | 5.57/6 (92.8%) |
| ALU utilization | 10.68/12 (89%) |

### Cycle Distribution

| Utilization | Cycles | Percentage |
|-------------|--------|------------|
| 6 VALU (max) | 1,123 | 86.1% |
| 5 VALU | 40 | 3.1% |
| 4 VALU | 32 | 2.5% |
| 3 VALU | 32 | 2.5% |
| 2 VALU | 36 | 2.8% |
| 1 VALU | 33 | 2.5% |
| 0 VALU | 9 | 0.7% |

### What Didn't Work

1. **Inline store emission**: Emitting stores immediately after each block's final round made things 12 cycles WORSE (fragmented stores created scheduling conflicts)

2. **Aggressive tiling**: Larger group_size/round_tile caused scratch overflow

3. **Different tile parameters**: Sweep of gs={14-20}, rt={10-16} confirmed gs=17, rt=13 is optimal

## Detailed Analysis (Session 4)

### Bundle Distribution
| (VALU, LOAD, STORE) | Count | Description |
|---------------------|-------|-------------|
| (3, 0, 0) | 982 | Hash stage2 without loads (WASTE) |
| (6, 2, 0) | 812 | Hash stage1 with prefetch (optimal) |
| (3, 2, 0) | 812 | Hash stage2 with prefetch |
| (6, 0, 0) | 434 | Various 6-op bundles |
| (0, 1, 0) | 211 | Load-only bundles |
| (2, 0, 0) | 198 | 2-op bundles |

### VALU Utilization
- Total VALU ops: 13,779
- Total bundles: 3,946
- VALU utilization: 58.2% (wasting 6,423 VALU slots)
- Low-util bundles (1-3 VALU): 1,994 bundles

### Theoretical Limits
- **VALU-bound minimum**: 13,312 ops / 6 = 2,219 cycles
- **Load-bound minimum**: 3,586 loads / 2 = 1,793 cycles
- **Actual**: 3,946 cycles

### Tree Structure Insight
Indices follow predictable patterns after broadcast rounds:
| Round | Unique Indices | Type |
|-------|----------------|------|
| 0 | 1 | broadcast (all at 0) |
| 1 | 2 | speculative ({1,2}) |
| 2 | 4 | speculative ({3-6}) |
| 3-10 | 8-224 | normal (diverging) |
| 11 | 1 | broadcast (all wrap to 0) |
| 12 | 2 | speculative ({1,2}) |
| 13-15 | 4-16 | speculative |

### Speculative Execution Potential
For rounds 1,2,12,13,14,15: preload unique tree values, use arithmetic selection.
- Load savings: 1,500 loads (41.8% reduction)
- VALU overhead: 576-1,248 ops (depends on selection complexity)
- **Net savings estimate: ~540-654 cycles**

## Target Thresholds
| Target | Cycles | Speedup | Status |
|--------|--------|---------|--------|
| Baseline | 147,734 | 1× | |
| Opus 4 many hours | <2,164 | 68× | ✓ |
| Opus 4.5 casual | <1,790 | 83× | ✓ |
| Sonnet 4.5 many hours | <1,548 | 95× | ✓ |
| Opus 4.5 2hr | <1,579 | 94× | ✓ |
| Opus 4.5 11hr | <1,487 | 99× | ✓ |
| Opus 4.5 improved | <1,363 | 108× | ✓ |
| **ACHIEVED** | **1,305** | **113.2×** | **✓** |
| Theoretical minimum | 1,212 | 121.9× | (92.9% achieved) |

## Current Performance
- Cycle count: 1,305
- Speedup: 113.2×
- Status: **COMPLETE** - Exceeds all target thresholds

## Theoretical Limits Analysis

The final solution is **VALU-bound**, not memory-bound:

| Resource | Total Ops | Slots/Cycle | Minimum Cycles |
|----------|-----------|-------------|----------------|
| VALU | 7,267 | 6 | **1,212** ← bottleneck |
| ALU | ~14,000 | 12 | 1,167 |
| Load | ~1,800 | 2 | 900 |
| Store | 64 | 2 | 32 |

### Why 93 Cycles Above Minimum?

The 1,305 - 1,212 = 93 cycle gap is structural overhead:

1. **Startup ramp** (~33 cycles): Init broadcasts, constant loading
2. **Drain phase** (~25 cycles): Final stores, pipeline emptying  
3. **Scheduling gaps** (~35 cycles): Dependencies forcing serialization

### Further Optimization Possibilities

To reduce further would require:
1. **Algorithmic changes**: Reduce total VALU ops (hash is fixed by problem)
2. **Globally optimal scheduler**: Current greedy scheduler is local-optimal
3. **Speculative execution**: Risk incorrect results for speed

At 92.9% of theoretical minimum, additional gains are marginal.

## Analysis Tools

Three tools in `scripts/` for understanding and optimizing VLIW kernels:

### 1. Bottleneck Detector (`scripts/bottleneck_detector.py`)

Determines whether kernel is VALU-bound, memory-bound, etc. and provides actionable recommendations.

```bash
python scripts/bottleneck_detector.py
```

Output:
- Operation counts per engine
- Theoretical minimum cycles per engine
- Identifies bottleneck (highest minimum)
- Efficiency percentage
- Specific recommendations based on bottleneck type

### 2. Profiler (`scripts/profiler.py`)

Per-cycle utilization breakdown with histograms and phase analysis.

```bash
python scripts/profiler.py                    # Summary + init/drain phases
python scripts/profiler.py --histogram        # Add utilization histograms
python scripts/profiler.py --bubbles          # Find low-utilization cycles
python scripts/profiler.py --phase steady     # Analyze middle section
```

Output:
- Average utilization per engine
- Utilization distribution histogram
- Init phase breakdown (first 50 cycles)
- Drain phase breakdown (last 50 cycles)
- Scheduling bubbles (cycles with VALU <= 3)

### 3. Schedule Visualizer (`scripts/visualize_schedule.py`)

ASCII visualization of VLIW bundles showing gaps and patterns.

```bash
python scripts/visualize_schedule.py --gaps           # Find contiguous low-util regions
python scripts/visualize_schedule.py --start 0 --end 100  # Visualize cycle range
python scripts/visualize_schedule.py --compact --all  # Full schedule, compact view
```

Output:
- ASCII bars showing per-cycle utilization
- Gap analysis (contiguous regions of low VALU)
- Position breakdown (init vs steady vs drain)

### Recommended Workflow

1. **Start with bottleneck detector** - Know what you're optimizing for
2. **Run profiler with --histogram** - Understand utilization distribution
3. **Run visualizer with --gaps** - Find where cycles are wasted
4. **Focus on largest gaps** - Init/drain phases often have most waste

### Example Session

```bash
$ python scripts/bottleneck_detector.py
DIAGNOSIS: VALU-BOUND
Theoretical minimum: 1212 cycles
Actual: 1305 cycles
Efficiency: 92.9%

$ python scripts/visualize_schedule.py --gaps
Total gap cycles: 110
  Init phase (first 50): 30 cycles
  Drain phase (last 50): 46 cycles
  Steady state: 0 gaps
```

This tells us: 76 of 93 wasted cycles are in init/drain phases. Steady state is optimal.

## External References
- Designing a SIMD Algorithm from Scratch: https://mcyoung.xyz/2023/11/27/simd-base64/
- AVX-512 Multi-hash Computation (Intel): https://networkbuilders.intel.com/docs/networkbuilders/intel-avx-512-ultra-parallelized-multi-hash-computation-for-data-streaming-workloads-technology-guide-1693301077.pdf
- Software Pipelining (Lam 1988): https://dl.acm.org/doi/10.1145/960116.54022
- Iron Law of Performance: https://blog.codingconfessions.com/p/one-law-to-rule-all-code-optimizations

## Known Notes
- Pre-existing lint warning: unused import `defaultdict` in `perf_takehome.py`.
