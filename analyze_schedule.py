"""Analyze scheduler output to find optimization opportunities."""

from collections import defaultdict
from problem import SLOT_LIMITS, VLEN, HASH_STAGES
from perf_takehome import KernelBuilder, _schedule_slots


def analyze_schedule():
    """Analyze the scheduled instructions for slot utilization."""
    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)

    # Count instructions before scheduling
    init_cycles = 0
    body_cycles = 0
    in_body = False

    # Analyze each cycle
    engine_usage = defaultdict(list)  # engine -> list of usage per cycle
    total_cycles = len(kb.instrs)

    for cycle_idx, cycle in enumerate(kb.instrs):
        if "flow" in cycle and any(op[0] == "pause" for op in cycle.get("flow", [])):
            in_body = True
            continue

        for engine, ops in cycle.items():
            engine_usage[engine].append(len(ops))

    print("=" * 60)
    print("SCHEDULER ANALYSIS")
    print("=" * 60)
    print(f"\nTotal cycles: {total_cycles}")
    print(f"\nSlot limits: {SLOT_LIMITS}")

    print("\n" + "-" * 40)
    print("ENGINE UTILIZATION:")
    print("-" * 40)

    for engine in ["alu", "valu", "load", "store", "flow"]:
        if engine in engine_usage:
            usages = engine_usage[engine]
            limit = SLOT_LIMITS[engine]
            total_used = sum(usages)
            max_possible = len(usages) * limit
            cycles_with_ops = len([u for u in usages if u > 0])

            print(f"\n{engine.upper()} (limit={limit}):")
            print(f"  Total ops: {total_used}")
            print(f"  Cycles with ops: {cycles_with_ops}")
            print(f"  Avg ops/cycle: {total_used/cycles_with_ops:.2f}" if cycles_with_ops > 0 else "  Avg ops/cycle: N/A")
            print(f"  Utilization: {total_used/max_possible*100:.1f}% (of max possible)")

            # Distribution
            dist = defaultdict(int)
            for u in usages:
                dist[u] += 1
            print(f"  Distribution: {dict(sorted(dist.items()))}")


def analyze_body_slots():
    """Analyze just the body slots before scheduling."""
    kb = KernelBuilder()

    # Manually extract body slots by building kernel
    group_size = 17
    round_tile = 13
    forest_height = 10
    batch_size = 256
    rounds = 16
    n_blocks = batch_size // VLEN

    # Count operations by type
    op_counts = defaultdict(int)

    # Per round per block analysis
    valu_per_round = 0
    alu_per_round = 0
    load_per_round = 0
    flow_per_round = 0

    # Count for different levels
    for level in range(forest_height + 1):
        valu_ops = 0
        alu_ops = 0
        load_ops = 0
        flow_ops = 0

        if level == 0:
            alu_ops += 8  # emit_xor
        elif level == 1:
            valu_ops += 1  # &
            flow_ops += 1  # vselect
            alu_ops += 8  # emit_xor
        elif level == 2:
            valu_ops += 3  # -, &, &
            flow_ops += 3  # 3 vselects
            alu_ops += 8  # emit_xor
        elif level == 3:
            valu_ops += 5  # -, &, &, -, &
            flow_ops += 7  # 7 vselects
            alu_ops += 8  # emit_xor
        else:  # level 4+
            alu_ops += 8  # address calc
            load_ops += 8  # gather
            alu_ops += 8  # emit_xor

        # Hash: 6 stages
        # Check which stages use multiply_add
        hash_valu = 0
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                hash_valu += 1  # multiply_add
            else:
                hash_valu += 3  # 3 ops
        valu_ops += hash_valu

        # idx update (for non-leaf levels)
        if level != forest_height:
            alu_ops += 16  # 8 & + 8 +
            valu_ops += 1  # multiply_add
        else:
            valu_ops += 1  # reset to 0

        print(f"\nLevel {level}:")
        print(f"  VALU: {valu_ops}, ALU: {alu_ops}, Load: {load_ops}, Flow: {flow_ops}")

    # Count level occurrences in 16 rounds
    print("\n" + "-" * 40)
    print("LEVEL FREQUENCY IN 16 ROUNDS:")
    print("-" * 40)
    level_counts = defaultdict(int)
    for rnd in range(16):
        level = rnd % (forest_height + 1)
        level_counts[level] += 1

    for level in range(forest_height + 1):
        print(f"Level {level}: {level_counts[level]} times")


def analyze_dependency_chains():
    """Analyze dependency chains in hash computation."""
    print("\n" + "=" * 60)
    print("HASH DEPENDENCY CHAIN ANALYSIS")
    print("=" * 60)

    print("\nHASH_STAGES:")
    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        can_ma = op1 == "+" and op2 == "+" and op3 == "<<"
        print(f"  Stage {i}: {op1} {val1:10x}, {op2}, {op3} {val3} -> {'multiply_add' if can_ma else '3 ops'}")

    # Count multiply_add eligible stages
    ma_count = sum(1 for op1, _, op2, op3, _ in HASH_STAGES
                   if op1 == "+" and op2 == "+" and op3 == "<<")
    other_count = len(HASH_STAGES) - ma_count

    print(f"\nMultiply_add stages: {ma_count}")
    print(f"3-op stages: {other_count}")
    print(f"Total VALU ops per hash: {ma_count + other_count * 3}")

    print("\nDependency chain per hash:")
    print("  val -> stage0 -> stage1 -> ... -> stage5 -> val'")
    print("  This is a 6-deep dependency chain!")
    print("  With multiply_add: some stages are 1 VALU op")
    print("  Without: stage needs 3 VALU ops but tmp1/tmp2 can overlap")


def estimate_ideal_cycles():
    """Estimate ideal cycle count with perfect scheduling."""
    print("\n" + "=" * 60)
    print("IDEAL CYCLE ESTIMATION")
    print("=" * 60)

    n_blocks = 32
    rounds = 16
    forest_height = 10

    # Count total operations
    total_valu = 0
    total_alu = 0
    total_load = 0
    total_flow = 0

    for rnd in range(rounds):
        level = rnd % (forest_height + 1)

        for block in range(n_blocks):
            if level == 0:
                total_alu += 8  # XOR
            elif level == 1:
                total_valu += 1
                total_flow += 1
                total_alu += 8
            elif level == 2:
                total_valu += 3
                total_flow += 3
                total_alu += 8
            elif level == 3:
                total_valu += 5
                total_flow += 7
                total_alu += 8
            else:  # level 4+
                total_alu += 16  # addr + XOR
                total_load += 8

            # Hash (count multiply_add eligible)
            ma_count = sum(1 for op1, _, op2, op3, _ in HASH_STAGES
                           if op1 == "+" and op2 == "+" and op3 == "<<")
            other_count = len(HASH_STAGES) - ma_count
            total_valu += ma_count + other_count * 3

            # idx update
            if level != forest_height:
                total_alu += 16
                total_valu += 1
            else:
                total_valu += 1

    # Load/store for init and final
    total_load += n_blocks * 2  # vload idx/val
    total_store = n_blocks  # vstore val

    print(f"\nTotal operations (body only):")
    print(f"  VALU: {total_valu}")
    print(f"  ALU:  {total_alu}")
    print(f"  Load: {total_load}")
    print(f"  Store: {total_store}")
    print(f"  Flow: {total_flow}")

    # Ideal cycles (assuming perfect parallelism)
    valu_cycles = total_valu / SLOT_LIMITS["valu"]
    alu_cycles = total_alu / SLOT_LIMITS["alu"]
    load_cycles = total_load / SLOT_LIMITS["load"]
    store_cycles = total_store / SLOT_LIMITS["store"]
    flow_cycles = total_flow / SLOT_LIMITS["flow"]

    print(f"\nIdeal cycles per engine (no dependencies):")
    print(f"  VALU: {valu_cycles:.1f} cycles ({SLOT_LIMITS['valu']} slots)")
    print(f"  ALU:  {alu_cycles:.1f} cycles ({SLOT_LIMITS['alu']} slots)")
    print(f"  Load: {load_cycles:.1f} cycles ({SLOT_LIMITS['load']} slots)")
    print(f"  Store: {store_cycles:.1f} cycles ({SLOT_LIMITS['store']} slots)")
    print(f"  Flow: {flow_cycles:.1f} cycles ({SLOT_LIMITS['flow']} slots)")

    print(f"\nBottleneck: VALU ({valu_cycles:.1f} cycles)")
    print(f"Current actual: 1307 cycles")
    print(f"Efficiency: {valu_cycles/1307*100:.1f}%")


def analyze_flow_valu_overlap():
    """Analyze overlap between Flow and VALU operations."""
    print("\n" + "=" * 60)
    print("FLOW/VALU OVERLAP ANALYSIS")
    print("=" * 60)

    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)

    # Categorize cycles
    flow_only = 0  # Flow ops but VALU idle or underutilized
    valu_only = 0  # VALU ops but no Flow
    both_full = 0  # Both Flow and VALU have ops
    neither = 0

    flow_with_valu_underutil = 0  # Flow + VALU < 6

    cycle_details = []

    for cycle_idx, cycle in enumerate(kb.instrs):
        flow_ops = len(cycle.get("flow", []))
        valu_ops = len(cycle.get("valu", []))
        alu_ops = len(cycle.get("alu", []))
        load_ops = len(cycle.get("load", []))

        # Skip pause cycles
        if "flow" in cycle and any(op[0] == "pause" for op in cycle.get("flow", [])):
            continue

        if flow_ops > 0 and valu_ops == 0:
            flow_only += 1
        elif flow_ops == 0 and valu_ops > 0:
            valu_only += 1
        elif flow_ops > 0 and valu_ops > 0:
            both_full += 1
            if valu_ops < 6:
                flow_with_valu_underutil += 1
        else:
            neither += 1

        if flow_ops > 0:
            cycle_details.append({
                "cycle": cycle_idx,
                "flow": flow_ops,
                "valu": valu_ops,
                "alu": alu_ops,
                "load": load_ops
            })

    print(f"\nCycle categorization:")
    print(f"  Flow only (VALU=0): {flow_only} cycles")
    print(f"  VALU only (Flow=0): {valu_only} cycles")
    print(f"  Both Flow+VALU: {both_full} cycles")
    print(f"    - VALU underutilized (<6): {flow_with_valu_underutil}")
    print(f"    - VALU full (=6): {both_full - flow_with_valu_underutil}")
    print(f"  Neither: {neither} cycles")

    print(f"\nTotal cycles with Flow: {flow_only + both_full}")
    print(f"Wasted VALU slots when Flow active: {flow_only * 6 + flow_with_valu_underutil * 3} (approx)")

    # Sample of flow cycles
    print("\n" + "-" * 40)
    print("SAMPLE: First 20 cycles with Flow ops")
    print("-" * 40)
    for detail in cycle_details[:20]:
        print(f"  Cycle {detail['cycle']:4d}: Flow={detail['flow']}, VALU={detail['valu']}, ALU={detail['alu']}, Load={detail['load']}")

    # Analyze consecutive Flow cycles
    print("\n" + "-" * 40)
    print("CONSECUTIVE FLOW CYCLES (potential bottleneck)")
    print("-" * 40)

    consecutive_runs = []
    current_run = 0
    for cycle_idx, cycle in enumerate(kb.instrs):
        flow_ops = len(cycle.get("flow", []))
        if "flow" in cycle and any(op[0] == "pause" for op in cycle.get("flow", [])):
            continue
        if flow_ops > 0:
            current_run += 1
        else:
            if current_run > 0:
                consecutive_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        consecutive_runs.append(current_run)

    if consecutive_runs:
        print(f"  Max consecutive Flow cycles: {max(consecutive_runs)}")
        print(f"  Avg consecutive Flow cycles: {sum(consecutive_runs)/len(consecutive_runs):.1f}")
        print(f"  Number of Flow runs: {len(consecutive_runs)}")

        # Distribution
        from collections import Counter
        dist = Counter(consecutive_runs)
        print(f"  Distribution of run lengths: {dict(sorted(dist.items()))}")


def analyze_dependency_stalls():
    """Analyze where dependencies cause stalls."""
    print("\n" + "=" * 60)
    print("DEPENDENCY STALL ANALYSIS")
    print("=" * 60)

    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)

    # Find cycles where VALU is underutilized
    underutil_cycles = []
    for cycle_idx, cycle in enumerate(kb.instrs):
        valu_ops = len(cycle.get("valu", []))
        if 0 < valu_ops < 6:
            underutil_cycles.append((cycle_idx, valu_ops, cycle))

    print(f"\nCycles with VALU underutilization (0 < ops < 6): {len(underutil_cycles)}")

    # Analyze what other engines are doing in these cycles
    other_engine_activity = defaultdict(int)
    for cycle_idx, valu_ops, cycle in underutil_cycles:
        for engine in ["alu", "load", "store", "flow"]:
            if cycle.get(engine):
                other_engine_activity[engine] += 1

    print(f"\nOther engine activity in VALU-underutilized cycles:")
    for engine, count in sorted(other_engine_activity.items()):
        print(f"  {engine}: {count} cycles")

    # Analyze distribution across the schedule
    total_cycles = len(kb.instrs)
    init_end = 0
    for i, cycle in enumerate(kb.instrs):
        if "flow" in cycle and any(op[0] == "pause" for op in cycle.get("flow", [])):
            init_end = i
            break

    init_underutil = [c for c, v, _ in underutil_cycles if c <= init_end]
    body_underutil = [c for c, v, _ in underutil_cycles if c > init_end]

    print(f"\nDistribution of VALU underutilization:")
    print(f"  Init phase (cycles 0-{init_end}): {len(init_underutil)} cycles")
    print(f"  Body phase (cycles {init_end+1}-{total_cycles}): {len(body_underutil)} cycles")

    # Analyze body underutilization in more detail
    if body_underutil:
        body_start = init_end + 1
        body_end = total_cycles

        # Find where in the body these occur
        early_body = [c for c in body_underutil if c < body_start + 100]
        mid_body = [c for c in body_underutil if body_start + 100 <= c < body_end - 100]
        late_body = [c for c in body_underutil if c >= body_end - 100]

        print(f"\n  Body breakdown:")
        print(f"    Early (first 100 cycles): {len(early_body)}")
        print(f"    Middle: {len(mid_body)}")
        print(f"    Late (last 100 cycles): {len(late_body)}")

    # Sample
    print("\nSample of VALU-underutilized cycles:")
    for cycle_idx, valu_ops, cycle in underutil_cycles[:10]:
        engines_active = [e for e in ["alu", "valu", "load", "store", "flow"] if cycle.get(e)]
        print(f"  Cycle {cycle_idx}: VALU={valu_ops}, active engines: {engines_active}")


def analyze_valu_saturation_gaps():
    """Find gaps where VALU could be doing more work."""
    print("\n" + "=" * 60)
    print("VALU SATURATION GAP ANALYSIS")
    print("=" * 60)

    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)

    # Find the pause that separates init from body
    init_end = 0
    for i, cycle in enumerate(kb.instrs):
        if "flow" in cycle and any(op[0] == "pause" for op in cycle.get("flow", [])):
            init_end = i
            break

    # Analyze body only
    body_cycles = kb.instrs[init_end+1:]

    # Find consecutive runs of VALU < 6
    gaps = []
    current_gap = []

    for i, cycle in enumerate(body_cycles):
        valu_ops = len(cycle.get("valu", []))
        if valu_ops < 6:
            current_gap.append((i + init_end + 1, valu_ops))
        else:
            if current_gap:
                gaps.append(current_gap)
            current_gap = []

    if current_gap:
        gaps.append(current_gap)

    # Filter significant gaps (length > 3)
    sig_gaps = [g for g in gaps if len(g) > 3]

    print(f"\nTotal gaps (consecutive VALU < 6): {len(gaps)}")
    print(f"Significant gaps (length > 3): {len(sig_gaps)}")

    if sig_gaps:
        print(f"\nLargest gaps:")
        sorted_gaps = sorted(sig_gaps, key=len, reverse=True)[:5]
        for gap in sorted_gaps:
            start_cycle = gap[0][0]
            length = len(gap)
            avg_valu = sum(v for _, v in gap) / length
            print(f"  Cycles {start_cycle}-{start_cycle+length-1}: length={length}, avg VALU={avg_valu:.1f}")

    # Calculate total wasted VALU slots in body
    body_wasted = sum(6 - len(cycle.get("valu", [])) for cycle in body_cycles if len(cycle.get("valu", [])) < 6)
    print(f"\nTotal wasted VALU slots in body: {body_wasted}")
    print(f"Equivalent to: {body_wasted / 6:.1f} cycles of VALU work")


def analyze_tail_region():
    """Analyze the tail region (cycles 1250-1306) in detail."""
    print("\n" + "=" * 60)
    print("TAIL REGION ANALYSIS (Cycles 1250-1306)")
    print("=" * 60)

    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)

    # Find init end
    init_end = 0
    for i, cycle in enumerate(kb.instrs):
        if "flow" in cycle and any(op[0] == "pause" for op in cycle.get("flow", [])):
            init_end = i
            break

    total_cycles = len(kb.instrs)
    tail_start = max(init_end + 1, total_cycles - 60)

    print(f"\nTail region: cycles {tail_start} to {total_cycles - 1}")
    print(f"\nPer-cycle breakdown:")
    print("-" * 70)
    print(f"{'Cycle':>6} | {'VALU':>4} | {'ALU':>4} | {'Load':>4} | {'Store':>5} | {'Flow':>4} | Notes")
    print("-" * 70)

    for cycle_idx in range(tail_start, total_cycles):
        if cycle_idx >= len(kb.instrs):
            break
        cycle = kb.instrs[cycle_idx]

        valu = len(cycle.get("valu", []))
        alu = len(cycle.get("alu", []))
        load = len(cycle.get("load", []))
        store = len(cycle.get("store", []))
        flow = len(cycle.get("flow", []))

        # Determine what's happening
        notes = []
        if store > 0:
            notes.append("STORE")
        if load > 0:
            notes.append("gather")
        if flow > 0:
            notes.append("vselect")
        if valu == 6:
            notes.append("VALU_FULL")
        elif valu > 0:
            notes.append(f"valu_partial")

        print(f"{cycle_idx:>6} | {valu:>4} | {alu:>4} | {load:>4} | {store:>5} | {flow:>4} | {', '.join(notes)}")

    # Summary stats for tail
    tail_cycles = kb.instrs[tail_start:total_cycles]
    total_valu = sum(len(c.get("valu", [])) for c in tail_cycles)
    total_alu = sum(len(c.get("alu", [])) for c in tail_cycles)
    total_load = sum(len(c.get("load", [])) for c in tail_cycles)
    total_store = sum(len(c.get("store", [])) for c in tail_cycles)
    total_flow = sum(len(c.get("flow", [])) for c in tail_cycles)

    print("-" * 70)
    print(f"\nTail summary ({len(tail_cycles)} cycles):")
    print(f"  VALU: {total_valu} ops (avg {total_valu/len(tail_cycles):.1f}/cycle, max 6)")
    print(f"  ALU:  {total_alu} ops (avg {total_alu/len(tail_cycles):.1f}/cycle, max 12)")
    print(f"  Load: {total_load} ops (avg {total_load/len(tail_cycles):.1f}/cycle, max 2)")
    print(f"  Store: {total_store} ops (avg {total_store/len(tail_cycles):.1f}/cycle, max 2)")
    print(f"  Flow: {total_flow} ops (avg {total_flow/len(tail_cycles):.1f}/cycle, max 1)")

    cycles_with_store = sum(1 for c in tail_cycles if c.get("store"))
    cycles_valu_full = sum(1 for c in tail_cycles if len(c.get("valu", [])) == 6)
    cycles_valu_empty = sum(1 for c in tail_cycles if len(c.get("valu", [])) == 0)

    print(f"\n  Cycles with store: {cycles_with_store}")
    print(f"  Cycles VALU full (=6): {cycles_valu_full}")
    print(f"  Cycles VALU empty (=0): {cycles_valu_empty}")


def analyze_block_completion():
    """Analyze when each block completes its last VALU write."""
    print("\n" + "=" * 60)
    print("BLOCK COMPLETION ANALYSIS")
    print("=" * 60)

    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)

    n_blocks = 32
    VLEN = 8

    # Find init end
    init_end = 0
    for i, cycle in enumerate(kb.instrs):
        if "flow" in cycle and any(op[0] == "pause" for op in cycle.get("flow", [])):
            init_end = i
            break

    # Track last write to each block's val_vec
    # val_base is allocated after idx_base
    # We need to find the actual addresses
    # From perf_takehome.py: idx_base and val_base are allocated in scratch

    # For simplicity, let's track by looking at VALU destination addresses
    # val_base starts at some offset, each block is VLEN apart

    # Actually, let's track all VALU writes and group by destination range
    block_last_write = {}  # block_id -> last_cycle

    # We need to figure out val_base. Looking at the code:
    # idx_base = self.alloc_scratch("idx_scratch", batch_size)  # 256 elements
    # val_base = self.alloc_scratch("val_scratch", batch_size)  # 256 elements

    # Let's scan for VALU ops that write to addresses in blocks
    # We'll use a heuristic: look for multiply_add ops (hash output) targeting consecutive ranges

    for cycle_idx, cycle in enumerate(kb.instrs):
        if cycle_idx <= init_end:
            continue

        for op in cycle.get("valu", []):
            if len(op) >= 2:
                dest = op[1]  # destination address
                # Estimate which block this belongs to
                # This is approximate - we're looking for patterns
                block_id = dest // VLEN
                if block_id not in block_last_write or cycle_idx > block_last_write[block_id]:
                    block_last_write[block_id] = cycle_idx

    # Filter to likely val_vec blocks (there should be 32 of them in a certain range)
    # Sort by block_id and find the 32 consecutive blocks with highest last_write values

    if not block_last_write:
        print("No VALU writes found")
        return

    # Find blocks and their last write cycles
    sorted_blocks = sorted(block_last_write.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 20 blocks by last VALU write cycle:")
    print("-" * 40)
    for block_id, last_cycle in sorted_blocks[:20]:
        print(f"  Block {block_id:3d} (addr {block_id*VLEN:4d}-{block_id*VLEN+VLEN-1:4d}): last write at cycle {last_cycle}")

    # Analyze spread
    all_last_cycles = list(block_last_write.values())
    if all_last_cycles:
        print(f"\nLast write cycle statistics:")
        print(f"  Min: {min(all_last_cycles)}")
        print(f"  Max: {max(all_last_cycles)}")
        print(f"  Spread: {max(all_last_cycles) - min(all_last_cycles)} cycles")

        # Distribution in last 100 cycles
        total_cycles = len(kb.instrs)
        late_blocks = [b for b, c in block_last_write.items() if c >= total_cycles - 100]
        very_late_blocks = [b for b, c in block_last_write.items() if c >= total_cycles - 50]

        print(f"\n  Blocks finishing in last 100 cycles: {len(late_blocks)}")
        print(f"  Blocks finishing in last 50 cycles: {len(very_late_blocks)}")


if __name__ == "__main__":
    analyze_body_slots()
    analyze_dependency_chains()
    estimate_ideal_cycles()
    print("\n")
    analyze_schedule()
    analyze_flow_valu_overlap()
    analyze_dependency_stalls()
    analyze_valu_saturation_gaps()
    analyze_tail_region()
    analyze_block_completion()
