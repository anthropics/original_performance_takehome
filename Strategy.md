# Strategy Log

This is a running log of optimization ideas we tried, what changed, and the
approximate cycle impact. All cycle numbers refer to the default workload
`forest_height=10, rounds=16, batch_size=256` unless noted.

## Baseline
- Scalar kernel (no VEC, no VLIW): ~147,734 cycles.

## VLIW scheduling
- Basic slot packing + dependency-aware scheduling: ~114,966 cycles.
- More aggressive scheduling (windowed ready set): no measurable change on current best kernels.

## Vectorization
- VEC (vector ops + scalar gather): large win vs baseline.
- VEC + hash-stage fusion (multiply_add on 3 stages): ~15,436 cycles (from ~16,974).

## Gather refactors
- `load_offset` gather with vector address compute: correct, small/no win.
- Prefetch-style overlap of next iteration addresses: ~13,916 cycles.
- VEC unroll x2 (interleave two vectors): ~7,258 cycles.
- VEC unroll x4: ~7,266 cycles (no win).
- VEC unroll x8: ~2,683 cycles (big win; near load limit).

## Hash interleaving
- Interleave hash stages across unrolled vectors:
  - VEC_UNROLL=2: ~7,246 cycles (small win vs 7,258).
  - VEC_UNROLL=8: ~2,683 cycles (with interleaving).

## Small-domain gather (exact)
- Round 0 (idx always 0): remove gather entirely (use forest[0] broadcast).
- Round 1 (idx in {1,2}): vselect between forest[1]/forest[2].
- With VEC_UNROLL=8 + SMALL_GATHER round 0/1: ~2,536 cycles (best so far).

## Per-value pipeline (rounds inside vector chunk)
- PER_VALUE_PIPE=1 (load idx/val once per chunk, run all rounds, store once):
  - ~2,180 cycles (large win; input loads/stores largely eliminated).
  - Bundles (out of ~2,180): alu 52, flow 1,058, load 1,986, store 61, valu 2,114.

## Parity + wrap arithmetic (vector)
- PARITY_AND=1 + ARITH_WRAP=1 (parity via AND, wrap via multiply):
  - ~2,108 cycles (beats 2,164 target).
  - Bundles (out of ~2,108): alu 49, flow 34, load 1,914, store 59, valu 2,044.

## Wrap skipping + scalar op2 (vector/ALU mix)
- SKIP_WRAP=1 (skip idx<n_nodes check except at max depth): ~2,024 cycles.
- SCALAR_OP2=1 (do op2 in ALU for non-fused stages) + SKIP_WRAP=1:
  - ~1,969 cycles.
  - Bundles (out of ~1,969): alu 1,392, flow 66, load 1,662, store 64, valu 1,896.

## Small-domain gather (attempts that didn’t help)
- Round 2 exact selector (idx 3..6): slower (rolled back).
- Round 3 exact selector (idx 7..14): slower (rolled back).
- Approximate 2-load round-2 split: incorrect (rolled back).

## Flow-to-VALU select (attempt)
- Replace `vselect` with VALU mask logic (VSELECT_VALU=1): ~2,676 cycles (worse; VALU saturates).
- Flow bundles dropped to ~2, but VALU bundles rose to ~2,612 (avg ~5.74/6).

## Scratch-resident inputs
- Keep inp indices/values in scratch across rounds (INP_SCRATCH=1):
  - Reduced load/store counts but **no cycle win** (bottleneck remains gather + hash).

## Double-buffer gather
- Attempted pipelined gather buffer (VEC_PIPE): no win; removed.

## Unroll > 8
- VEC_UNROLL=12: slightly worse (~2,567 cycles).
- VEC_UNROLL=16: incorrect due to scratch pressure (capped to 8).

## Bit-slicing feasibility
- Bit-slice cost estimate: ~1,280 bitwise ops per value per hash (too expensive to pursue).

## Summary (one try per section)
### Vectorization + VLIW + unroll=8
- Large win vs scalar; brought cycles down to the low 2K range.

### Per-value pipeline (PER_VALUE_PIPE)
- Large win by removing per-round input loads/stores.

### Parity via AND + wrap via multiply (PARITY_AND + ARITH_WRAP)
- Moderate win; reduced flow usage and lowered cycles further.

### Skip wrap except at max depth (SKIP_WRAP)
- Moderate win; removed redundant bounds checks in most rounds.

### Move op2 to ALU (SCALAR_OP2)
- Moderate win; reduced VALU pressure by shifting op2 to ALU lanes.

### Hash-stage fusion (multiply_add)
- Small win; fewer VALU slots in select stages.

### Hash interleaving across unroll
- Small win; improved slot packing.

### VSELECT_VALU path
- Regression; VALU saturated and cycles increased.

### INP_SCRATCH
- No win; reduced load/store counts but no cycle improvement.

### Double-buffer gather (VEC_PIPE)
- No win; complexity without cycle reduction.

### Unroll > 8
- Worse or incorrect; scratch pressure and correctness issues.

### Round 2/3 selector attempts
- Incorrect or slower; rolled back.

### Bit-slicing estimate
- Too expensive; not pursued.

### Current bottlenecks
- Gather loads + VALU remain near limits; flow largely removed. Further gains likely need fewer gathers or hash simplification.

## Current best settings
- `VEC=1 VLIW=1 VEC_UNROLL=8 SMALL_GATHER=1 PER_VALUE_PIPE=1 PARITY_AND=1 ARITH_WRAP=1 SKIP_WRAP=1 SCALAR_OP2=1`
- ~1,969 cycles (still above 1,300 target; below 2,164).

### Slot utilization and bundle counts
Engine | Avg/Max | Bundles (out of 1,969)
alu | 8.88 / 12 | 1,392
valu | 4.06 / 6 | 1,896
load | 1.92 / 2 | 1,662
store | 1.00 / 2 | 64
flow | 1.00 / 1 | 66
