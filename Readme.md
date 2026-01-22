Optimization Overview (High Level)

Goal
- Reduce total cycles for the VLIW/SIMD kernel by improving instruction density and balancing engine utilization.

What the current code does (conceptual optimizations)
1) Streamed scheduling / bundling
   - Treat each SIMD batch (VLEN=8) as an independent stream of tasks.
   - Each cycle, build a bundle by pulling from per-engine queues (load/valu/alu/flow), while enforcing RAW/WAW hazards and slot limits.
   - This increases per-cycle utilization by mixing independent streams.

2) Staggered stream starts
   - Offset the start time of each stream so their load-heavy phases do not align.
   - Keeps the load engine closer to its 2-slot limit and avoids long load bubbles.
   - Tuned via STAGGER_GROUPS/STEP (defaults currently 9/35).

3) Load scheduling policy
   - Prefer draining short bursts from one stream, then fill a second load slot from another stream if available.
   - Optionally spread loads across streams (LOAD_SPREAD) when beneficial.

4) Vectorized data path
   - Use vload/vstore and vector ALU (valu) to operate on 8 lanes at once.
   - Pre-broadcast scalar constants into vector registers to avoid repeated loads.

5) Hash micro-op fusion
   - Use multiply_add for shift-add stages in the hash to reduce op count.
   - Keeps the hash dominated by valu/alu ops instead of extra loads.

6) Cached early tree nodes
   - Preload/broadcast early tree nodes (root and depth-2 nodes).
   - Use vselect to choose among cached node values, cutting early load traffic.

7) Trace-driven tuning
   - Added trace instrumentation to measure per-cycle slot usage and engine utilization.
   - Used these stats to target load bottlenecks and scheduler staggering.

Notes
- N_CORES remains 1 as required.
- Environment knobs used for experimentation: STAGGER_GROUPS, STAGGER_STEP, STAGGER_MODE, STAGGER_LIST, STAGGER_PERIOD, LOAD_SPREAD, HASH_ADD_ALU, ALLOW_WAR.
