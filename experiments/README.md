# Experiments Directory

This directory contains scripts to test specific optimization theories for the kernel.

## Scripts

### 1. `batch_sweep.py`
Sweeps `BATCH` sizes (6, 8, 12, 16, 20, 24) to find the optimal scratch-pad utilization vs pipeline drain trade-off.
**Usage**: `python3 experiments/batch_sweep.py`

### 2. `baseline.py`
Snapshot of the current working `perf_takehome.py` (BATCH=16 + Small Tree Optimization).
**Performance**: ~3053 cycles.
**Usage**: `python3 experiments/baseline.py Tests.test_kernel_cycles`

### 3. `medium_tree_opt.py`
Implementation of "Medium Tree" optimization (Round 2 and 13). Replaces Gathers for indices 3,4,5,6 with Arithmetic Selection.
**Logic**:
- Uses `input_index` (3..6) to select from cached Constants (Tree values 3,4,5,6) instead of Load.
- Saves ~512 Loads per run (expected speedup ~250 cycles).
**Status**: Currently failing correctness checks on Round 1. Logic for parent/sibling selection needs debugging.

## Theory vs Practice
- **Baseline (BATCH=16, R1/12 Opt)**: 3053 cycles. 
- **Target**: 2164 cycles.
- **Medium Tree (R2/13)**: Should reduce cycles to ~2800.
- **Next Steps**: Debug Round 1 failure in `medium_tree_opt.py`. Likely a scratch layout or register reuse issue.
