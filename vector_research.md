# Research: Vectorization for Performance Takehome (VLEN=8 Platform)

## 1. Problem Characterization
The platform is a simulated VLIW/SIMD machine with tight constraints:
- **VLEN=8**: Matches AVX2-style width (for 32-bit floats/ints).
- **Bottlenecks**:
    - **Flow (1/cycle)**: `vselect` implies a blend/mux operation that consumes this scarce slot.
    - **Load (2/cycle)**: "Gathers" (indirect loads `mem[idx[i]]`) typically decompose into scalar loads or are very slow. Here they consume 1 Load slot per element? No, the guide says "8 gathers per vector = 4 cycles" (assuming 2 loads/cycle). This is a massive improvement over standard x86 where gathers are often 8+ uops. However, it's still the bottleneck compared to VALU (6/cycle).

## 2. Optimization Strategies

### A. Eliminating Flow Control (Predication)
Since `vselect` consumes the bottleneck Flow slot, we must convert control flow to data flow.
- **Arithmetic Selection**: Instead of `res = vselect(cond, a, b)`, use `res = (a & mask) | (b & ~mask)` or `res = b + ((a-b) & mask)`.
    - This uses **VALU** slots (6 available!) instead of Flow slots.
    - *Caveat*: Requires valid values for both paths (no segfaults on the "not taken" path).

### B. Minimizing Gathers (Data Layout)
Gathers are expensive. Can we use **unit-stride (contiguous) loads**?
- **Tree Layout**: If the data structure is a tree, a **Byte-Mapped** or **BFS/Level-Order** layout allows loading a cluster of nodes with a single `vload`.
    - *Classic BST*: Children at `2i` and `2i+1`.
    - *Block-based*: Store a small subtree (e.g., indices 1..15) in a single cache line (or contiguous block).
    - *Optimization*: If we need to traverse down, we can load a block of 8 nodes, perform comparisons in parallel (VALU), and compute the next index.

- **Hash Probing**:
    - **Linear Probing**: Just load a contiguous chunk (8 buckets). Check keys in parallel.
    - **Cuckoo Hashing**: Requires multiple lookups, likely gathers. Avoid if possible.
    - **SoA vs AoS**: Standard "Structure of Arrays" usually wins for SIMD. Store all Keys in one array, all Values in another. `vload` 8 keys at once.
    - **Vectorized Linear Probing (VLP)**:
        - Load a vector of keys and compare all simultaneously using SIMD.
        - Requires horizontal vectorization (bucketing N keys) or vertical vectorization.
        - **Horizontal Vectorization**: Structure hash table into "buckets" of N keys (where N=VLEN). Lookup checks one bucket in parallel using SIMD logical ops. This is frictionless if we ensure collisions are handled within buckets or chained.

### C. Software Pipelining (Modulo Scheduling)
The guide mentions memory latency ("Loads complete at end of cycle").
- **Double Buffering**: Load iteration `i+1` while processing iteration `i`.
- **Modulo Scheduling**:
    - Cycle `t`: Load `k[i+2]`, Compute `h[i+1]`, Store `res[i]`
    - Keeps all units (Load, VALU, Store) busy.
    - *Constraint*: Scratch space is limited (1536 words). Deep pipelines increase register pressure.
    - **Implementation**: Interleave "Gather" ops from Batch B+1 with "Hash" ops from Batch B. This hides the latency of gathers behind the compute-intensive hash stages.

### D. Algorithmic "Hacks"
- **Multi-key Lookup**: Is the problem looking up multiple independent keys?
    - If so, process 8 independent searches in parallel (classic SIMD approach).
    - *Divergence*: If searches finish at different times, use a "lane refill" or just masking which wastes slots.
    - *Better*: "Sorting" queries to group similar paths? (Overhead often too high).

## 3. Specific Architecture Quirks
- **Scratchpad as Register File**: 1536 words is large compared to ymm registers (usually 16x8=128 words). This suggests we can unroll loops aggressively and keep large lookup tables (e.g., precomputed constants) in scratch.

## 4. Suggested Experiments
1. **Arithmetic Select**: Replace *all* `vselect` with bitwise operations. Measure cycle impact.
2. **Blocked Hashing**: Change hash table logic (if allowed) to check 8 contiguous buckets.
3. **Speculative Loading**: Load common paths (e.g., if 90% of branches go left, load left child speculatively).
