# Strategy Log

This is a running log of optimization ideas we tried, what changed, and the
approximate cycle impact. All cycle numbers refer to the default workload
`forest_height=10, rounds=16, batch_size=256` unless noted.

## Baseline
- Scalar kernel (no VEC, no VLIW): ~147,734 cycles.

## VLIW scheduling
- Basic slot packing + dependency-aware scheduling: ~114,966 cycles.
- More aggressive scheduling (windowed ready set): no measurable change on current best kernels.
- Critical-path priority scheduling + opportunistic fill (deterministic):
  - ~1,576 cycles (from ~1,637) on current kernel.
  - Added optional flags: `disambiguate_mem` (affine mem keys) and `load_offset_reads_base` (ISA semantics); both left off by default.
- Weighted critical-path + slack tie-break (scheduler flags only):
  - `SCHED_WEIGHTED=1` regressed to ~1,647 cycles.
  - `SCHED_SLACK=1` no measurable change (~1,576 cycles).
- One-bundle lookahead repair (scheduler flag only):
  - `SCHED_REPAIR=1` improved to ~1,568 cycles.
- Expanded mem disambiguation (scheduler flag only):
  - `SCHED_MEM_DISAMBIG=1` improved to ~1,545 cycles (passes 1,548 threshold).
- Mem disambiguation + repair (scheduler flags only):
  - `SCHED_MEM_DISAMBIG=1 SCHED_REPAIR=1` improved to ~1,534 cycles.

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
- Depth 3 (idx in 7..14): preload forest[7..14], 8-way flow vselect; ~1,503 cycles (from ~1,534).
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
- Depth-2/3 no-gather selector (SMALL_GATHER_D2/SMALL_GATHER_D3): regressed; selection cost outweighed load savings.
- Depth-2 ALU selector (SMALL_GATHER_D2_ALU): regressed further.
- Duplicate-idx broadcast gather idea: not viable without predicated loads/control (ISA/scheduler limitation).
- Depth-3 flow-vselect selector: incorrect results (removed).
- Depth-2 flow selector (SMALL_GATHER_D2_FLOW): small win; cycles ~1,899 (down from 1,969) but still above targets.
- Depth-3 flow selector with range guard (SMALL_GATHER_D3_FLOW): correct but slow (~2,457 cycles); not worth using.
- Depth-3 VALU arithmetic selector (idx 7..14 with multiply_add): regressed (~1,738 cycles); rolled back.
- Depth-3 VALU mask selector (8-way, no flow/gather): regressed (~1,876 cycles); rolled back.

## Hash simplification (attempts)
- Scalarize XOR-stage op1 (SCALAR_XOR_OP1=1): regressed (~2,162 cycles); not useful.

## Wrap skipping (extra)
- MAX_DEPTH_ZERO=1: at max depth, skip parity/update and set idx=0 directly; small win (~1,873 cycles).
- SKIP_LAST_IDX=1: skip idx update on final round; no win (slightly worse, ~1,797 cycles).

## Index update simplification
- IDX_MADD=1: compute idx update as `idx = idx*2 + (val&1)` via multiply_add; ~1,807 cycles.
- IDX_DEPTH0_DIRECT=1: when depth==0, set idx directly to (val&1)+1; brings cycles down to ~1,777.

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
- VEC_UNROLL=16: works in current kernel; ~1,664 cycles (new best).
- Current kernel profile (unroll=20): ~1,650 cycles. Bundles (profile @ ~1,650): alu 1,249, flow 258, load 1,432, store 31, valu 1,556.

## Address setup (stride/pointer bump)
- Pointer-bump unrolled vector addresses by +VLEN instead of per-ui base constants:
  - ~1,637 cycles (improves ~1,650).

## Scheduler priority (critical path)
- Priority-based ready selection (longest-path) + opportunistic fill:
  - ~1,576 cycles (improves ~1,637).
  - PROFILE=1 bundles: alu 1,245, flow 258, load 1,356, store 26, valu 1,468.

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

### Address setup (pointer bump)
- Small win by reducing ALU address setup overhead and constant pressure.

### Priority scheduler (critical path)
- Small win by prioritizing critical ops and filling unused engine slots.

### Round 2/3 selector attempts
- Incorrect or slower; rolled back.

### Bit-slicing estimate
- Too expensive; not pursued.

### Current bottlenecks
- Gather loads + VALU remain near limits; flow largely removed. Further gains likely need fewer gathers or hash simplification.

## Current best settings
- Best path is now hardcoded (flags removed): VEC+VLIW, unroll=20, per‑value pipeline, depth‑0/1/2/3 small‑gather, parity via AND, idx update via multiply_add, depth‑0 direct idx, max‑depth idx=0.
- ~1,503 cycles with scheduler flags (`SCHED_MEM_DISAMBIG=1 SCHED_REPAIR=1`).

### Slot utilization and bundle counts
Engine | Avg/Max | Bundles (profile @ ~1,534 cycles)
alu | 9.87 / 12 | 1,249
valu | 4.67 / 6 | 1,478
load | 1.95 / 2 | 1,349
store | 1.07 / 2 | 30
flow | 1.00 / 1 | 258
