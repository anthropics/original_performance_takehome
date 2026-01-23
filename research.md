# Kernel Optimization Research Log

## Goal
Reduce simulated kernel cycles (submission harness) from the baseline 147,734 toward <2,000 cycles without modifying tests.

## Current Best (as of latest sweep)
- **Cycles:** ~2,847
- **Config:** unroll=6, load_interleave=2, schedule=stagewise
- **Notes:** Stable across recent sweeps; alternative interleaving/block grouping did not improve.

## Tooling Added
- `tools/kernel_stats.py`: instruction mix + slot utilization per engine.
- `tools/kernel_profiler.py`: segment/marker-based profiling and utilization.
- `tools/kernel_tuner.py`: small brute-force sweep for unroll/load/schedule.
- `tools/kernel_hybrid_tuner.py`: hybrid brute-force sweep for hash interleave, block grouping, gather strategy.
- `watch_trace.py` + `watch_trace.html`: Perfetto/trace viewer for instruction bundles.

## Optimizations Tried (and Outcomes)
### Vectorization and Slot Packing
- Vectorized kernel loop using SIMD (`vload`, `valu`, `vstore`) for the bulk of the batch.
- Packed VLIW slots with hazard-aware bundling to fill `alu/valu/load/store` engines.
- **Outcome:** Major cycle reduction vs scalar baseline.

### Double-Buffered Vector Pipelining
- Overlapped gather loads with hash compute using alternating buffer sets.
- Added interleaving of pending loads within compute.
- **Outcome:** Significant improvement; load engine better utilized.

### Unroll and Scheduling Tweaks
- Unroll factor adjustments: 2 -> 6 (best), 8 regressed.
- Load interleave count: 1, 2, 3 tested; **2 best**.
- Blockwise schedule (compute each block fully before next) was much worse.
- Block grouping (group size smaller than unroll) showed no improvement.

### Instruction Substitutions
- Replaced `flow vselect` with `valu multiply_add` using 0/1 masks to reduce flow slots.
- **Outcome:** Reduced flow pressure and improved packing.

### Hash and Gather Scheduling Experiments
- Hash-stage interleave patterns `(h1,h2,h3)` across valu ops (0..3 each): no improvement.
- Gather ordering: by buffer vs round-robin across buffers: no improvement.

## Brute-Force Sweep Summary
### Simple sweep (kernel_tuner)
- Unroll ∈ {4,6,8}, load_interleave ∈ {1,2,3}, schedule ∈ {stagewise,blockwise}
- **Best:** unroll=6, load_interleave=2, stagewise → 2,847 cycles

### Hybrid sweep (kernel_hybrid_tuner)
- Searched hash_interleave, block_group, gather_strategy
- **Result:** All variants held at 2,847 cycles; no improvement found.

## Observations
- Load engine is a bottleneck; compute can often hide load latency only with careful interleaving.
- Flow engine slots are expensive; removing `vselect` helped.
- Unroll=6 likely balances ILP with scratch pressure.

## Ideas Still To Do
### Scheduling / Pipelining
- **Modulo scheduling style search:** use a fixed initiation interval to overlap iterations more systematically (software pipelining concept).
- **Two-phase pipeline:** prefetch next iteration’s gather earlier (or prefetch deeper into future) to smooth load bursts.
- **Cross-round pipelining:** overlap the tail of one round with the start of the next when possible.

### Load/Gather Experiments
- **Lane-clustered gather:** issue a subset of lanes across all buffers, then the next subset (e.g., lanes 0..3 for all buffers, then 4..7).
- **Staggered gather per block:** alternate load ordering to reduce burstiness (e.g., A0,A1,B0,B1, ...).
- **Adaptive gather strategy:** choose ordering based on `unroll` or load_interleave.

### Instruction Mix / Dataflow
- **Reduce temporaries:** attempt to reuse temp registers inside hash stages to ease scratch pressure.
- **Bitmask wrap if possible:** if `n_nodes` is power-of-two, use bitmask instead of compare/select (needs validation).
- **Hash stage grouping:** attempt reordering of independent operations to better align with load interleaving (keep correctness).

### Search / Automation
- **Random search overnight:** extend hybrid tuner to thousands of trials (already supported).
- **Heuristic search:** simulated annealing / hill climb on scheduling knobs.
- **Micro-scheduler:** generate alternative VLIW packing orders for the same logical ops to improve slot utilization.

## Overnight Sweep Commands (Recommended)
Random hybrid sweep (5k trials):
```
python tools\kernel_hybrid_tuner.py --mode random --trials 5000 --output overnight_random.csv --gather-strategy by_buffer,round_robin
```
Full grid (longer):
```
python tools\kernel_hybrid_tuner.py --mode grid --unroll 4,5,6,7,8 --load-interleave 1,2,3 --hash-values 0,1,2,3 --block-groups 1,2,3,4,unroll --gather-strategy by_buffer,round_robin --output overnight_grid.csv
```

## References (Scheduling / Software Pipelining)
- Basic instruction scheduling + software pipelining overview (ramp-up/ramp-down, overlap iterations, modulo scheduling):
  https://www.lighterra.com/papers/basicinstructionscheduling/

- Documentation of this problem:
  https://trirpi.github.io/posts/anthropic-performance-takehome/
