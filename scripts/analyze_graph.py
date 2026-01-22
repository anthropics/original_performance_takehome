#!/usr/bin/env python3
"""Programmatically analyze the kernel execution graph."""

import sys

sys.path.insert(0, "/home/travers/brain-2/01-projects/original_performance_takehome")

from collections import defaultdict
from perf_takehome import KernelBuilder
from problem import VLEN, HASH_STAGES


def analyze_kernel():
    kb = KernelBuilder()
    kb.build_kernel(10, 1023, 256, 16)

    n_vectors = 256 // VLEN  # 32
    n_triples = n_vectors // 3  # 10
    remainder = n_vectors % 3  # 2

    print(
        f"Configuration: {n_vectors} vectors, {n_triples} triples, {remainder} remainder"
    )
    print(f"Total bundles: {len(kb.instrs)}")
    print()

    # Categorize bundles
    bundle_stats = defaultdict(
        lambda: {"count": 0, "valu_ops": 0, "load_ops": 0, "store_ops": 0}
    )

    # Track bundle index ranges for different phases
    phases = []
    current_phase = "init"
    phase_start = 0

    for i, bundle in enumerate(kb.instrs):
        valu = len(bundle.get("valu", []))
        load = len(bundle.get("load", []))
        store = len(bundle.get("store", []))
        flow = len(bundle.get("flow", []))

        # Detect phase transitions
        if flow > 0:
            if current_phase != "init":
                phases.append((current_phase, phase_start, i))
            current_phase = "pause" if i < 100 else "end"
            phase_start = i + 1

        bundle_stats[(valu, load, store)]["count"] += 1
        bundle_stats[(valu, load, store)]["valu_ops"] += valu
        bundle_stats[(valu, load, store)]["load_ops"] += load
        bundle_stats[(valu, load, store)]["store_ops"] += store

    # Final phase
    phases.append((current_phase, phase_start, len(kb.instrs)))

    print("=" * 60)
    print("BUNDLE TYPE DISTRIBUTION")
    print("=" * 60)
    print(f"{'(VALU,LOAD,STORE)':<20} {'Count':<10} {'VALU ops':<12} {'Load ops':<12}")
    print("-" * 60)

    total_valu = 0
    total_load = 0
    wasted_valu_slots = 0

    for key in sorted(bundle_stats.keys(), key=lambda x: -bundle_stats[x]["count"]):
        stats = bundle_stats[key]
        print(
            f"{str(key):<20} {stats['count']:<10} {stats['valu_ops']:<12} {stats['load_ops']:<12}"
        )
        total_valu += stats["valu_ops"]
        total_load += stats["load_ops"]
        # Wasted = bundles with <6 VALU ops
        if key[0] < 6 and key[0] > 0:
            wasted_valu_slots += stats["count"] * (6 - key[0])

    print("-" * 60)
    print(f"Total VALU ops: {total_valu}")
    print(f"Total Load ops: {total_load}")
    print(f"Wasted VALU slots: {wasted_valu_slots}")
    print(f"VALU utilization: {total_valu / (len(kb.instrs) * 6) * 100:.1f}%")
    print()

    # Analyze per-round structure
    print("=" * 60)
    print("ROUND STRUCTURE ANALYSIS")
    print("=" * 60)

    # Find the main loop start (after init section)
    main_start = None
    for i, bundle in enumerate(kb.instrs):
        if "flow" in bundle and ("pause",) in bundle.get("flow", []):
            main_start = i + 1
            break

    if main_start is None:
        print("Could not find main loop start")
        return

    # Estimate bundles per round
    # Total main loop bundles / 16 rounds
    main_end = len(kb.instrs)
    for i in range(len(kb.instrs) - 1, -1, -1):
        if "flow" in kb.instrs[i]:
            main_end = i
            break

    # Find store section
    store_start = None
    for i in range(main_end - 1, main_start, -1):
        if "store" in kb.instrs[i]:
            if store_start is None:
                store_start = i
        elif store_start is not None:
            store_start = i + 1
            break

    main_loop_bundles = (store_start or main_end) - main_start
    bundles_per_round = main_loop_bundles / 16

    print(f"Init section: bundles 0-{main_start - 1} ({main_start} bundles)")
    print(
        f"Main loop: bundles {main_start}-{(store_start or main_end) - 1} ({main_loop_bundles} bundles)"
    )
    print(f"Store section: bundles {store_start or 'N/A'}-{main_end - 1}")
    print(f"Average bundles per round: {bundles_per_round:.1f}")
    print()

    # Analyze theoretical minimum
    print("=" * 60)
    print("THEORETICAL ANALYSIS")
    print("=" * 60)

    # Per vector per round:
    # - XOR: 1 op
    # - Hash: 6 stages × 3 ops = 18 ops
    # - Index: AND(1) + ADD(1) + MAD(1) + CMP(1) + MUL(1) = 5 ops
    # - Writeback: 2 ops (copy to scratch)
    ops_per_vector = 1 + 18 + 5 + 2
    total_ops = n_vectors * 16 * ops_per_vector

    print(f"Ops per vector per round: {ops_per_vector}")
    print(f"Total VALU ops needed: {total_ops}")
    print(f"Minimum cycles (VALU-bound): {total_ops / 6:.0f}")
    print()

    # Load analysis
    # Broadcast rounds (0, 11): 1 load per round
    # Normal rounds: 8 loads per vector × n_vectors
    broadcast_loads = 2 * 1  # rounds 0, 11
    normal_loads = 14 * n_vectors * 8  # 14 normal rounds
    total_loads = broadcast_loads + normal_loads

    print(f"Broadcast round loads: {broadcast_loads}")
    print(f"Normal round loads: {normal_loads}")
    print(f"Total loads: {total_loads}")
    print(f"Minimum cycles (load-bound): {total_loads / 2:.0f}")
    print()

    # Identify optimization opportunities
    print("=" * 60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 60)

    # Count bundles by VALU utilization
    low_util_bundles = sum(
        stats["count"] for key, stats in bundle_stats.items() if 0 < key[0] < 4
    )
    med_util_bundles = sum(
        stats["count"] for key, stats in bundle_stats.items() if 4 <= key[0] < 6
    )
    high_util_bundles = sum(
        stats["count"] for key, stats in bundle_stats.items() if key[0] == 6
    )

    print(f"Low utilization (1-3 VALU): {low_util_bundles} bundles")
    print(f"Medium utilization (4-5 VALU): {med_util_bundles} bundles")
    print(f"High utilization (6 VALU): {high_util_bundles} bundles")
    print()

    # Identify patterns in low-util bundles
    print("Low-util bundle patterns:")
    for key, stats in sorted(bundle_stats.items(), key=lambda x: -x[1]["count"]):
        if 0 < key[0] < 4:
            print(f"  {key}: {stats['count']} bundles")
            # Sample some bundles of this type
            samples = []
            for i, b in enumerate(kb.instrs):
                v = len(b.get("valu", []))
                l = len(b.get("load", []))
                s = len(b.get("store", []))
                if (v, l, s) == key and len(samples) < 3:
                    samples.append((i, b))
            for idx, b in samples:
                ops = b.get("valu", [])
                print(f"    Bundle {idx}: {[op[0] for op in ops]}")


def analyze_speculative_potential():
    """Analyze potential savings from speculative execution."""
    print()
    print("=" * 60)
    print("SPECULATIVE EXECUTION ANALYSIS")
    print("=" * 60)

    n_vectors = 32

    # Rounds where indices are constrained
    constrained_rounds = {
        0: (1, "broadcast"),  # All idx=0
        1: (2, "speculative"),  # idx in {1,2}
        2: (4, "speculative"),  # idx in {3,4,5,6}
        11: (1, "broadcast"),  # All idx=0
        12: (2, "speculative"),
        13: (4, "speculative"),
        14: (8, "speculative"),
        15: (16, "speculative"),
    }

    print(f"Round analysis (16 rounds total):")
    print(f"{'Round':<8} {'Unique idx':<12} {'Type':<15} {'Loads needed':<15}")
    print("-" * 50)

    total_current_loads = 0
    total_speculative_loads = 0

    for rnd in range(16):
        if rnd in constrained_rounds:
            n_idx, rtype = constrained_rounds[rnd]
            if rtype == "broadcast":
                current_loads = 1
                spec_loads = 1
            else:
                current_loads = n_vectors * 8  # Current: full gather
                spec_loads = n_idx  # Speculative: just preload unique values
        else:
            current_loads = n_vectors * 8
            spec_loads = n_vectors * 8
            n_idx = "many"
            rtype = "normal"

        total_current_loads += current_loads
        total_speculative_loads += spec_loads

        print(f"{rnd:<8} {str(n_idx):<12} {rtype:<15} {current_loads} -> {spec_loads}")

    print("-" * 50)
    print(f"Total loads: {total_current_loads} -> {total_speculative_loads}")
    print(
        f"Load reduction: {total_current_loads - total_speculative_loads} ({(1 - total_speculative_loads / total_current_loads) * 100:.1f}%)"
    )
    print()

    # Cycle savings estimate
    load_cycles_saved = (total_current_loads - total_speculative_loads) / 2
    print(f"Estimated load cycles saved: {load_cycles_saved:.0f}")

    # But we add VALU ops for arithmetic selection
    # For 2-value speculation: 3 ops per vector (sub, mul, add)
    # For 4-value speculation: more complex selection
    spec_valu_overhead = 0
    for rnd, (n_idx, rtype) in constrained_rounds.items():
        if rtype == "speculative":
            if n_idx == 2:
                spec_valu_overhead += n_vectors * 3  # sub, mul, add
            elif n_idx == 4:
                spec_valu_overhead += n_vectors * 6  # More complex

    print(
        f"Speculative VALU overhead: {spec_valu_overhead} ops = {spec_valu_overhead / 6:.0f} cycles"
    )
    print(
        f"Net cycle change estimate: {-load_cycles_saved + spec_valu_overhead / 6:.0f}"
    )


if __name__ == "__main__":
    analyze_kernel()
    analyze_speculative_potential()
