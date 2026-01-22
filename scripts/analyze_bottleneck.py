#!/usr/bin/env python3
"""Deep analysis of the bottleneck and what would be needed for <1700 cycles."""

import sys

sys.path.insert(0, "/home/travers/brain-2/01-projects/original_performance_takehome")

from problem import VLEN, HASH_STAGES


def main():
    n_vectors = 256 // VLEN  # 32
    rounds = 16

    print("=" * 70)
    print("WHAT WOULD IT TAKE TO ACHIEVE <1700 CYCLES?")
    print("=" * 70)
    print()

    # Current state
    current_cycles = 3946
    target_cycles = 1700
    required_speedup = current_cycles / target_cycles

    print(f"Current: {current_cycles} cycles")
    print(f"Target: <{target_cycles} cycles")
    print(f"Required speedup: {required_speedup:.2f}x")
    print()

    # Absolute minimums
    print("=" * 70)
    print("ABSOLUTE THEORETICAL MINIMUMS")
    print("=" * 70)

    # VALU ops required (can't reduce without changing algorithm)
    ops_per_hash = 18  # 6 stages × 3 ops
    ops_per_round_per_vec = 1 + ops_per_hash + 5 + 2  # XOR + hash + index + writeback
    total_valu_ops = n_vectors * rounds * ops_per_round_per_vec
    min_valu_cycles = total_valu_ops / 6

    print(f"Total VALU ops: {total_valu_ops}")
    print(f"Min VALU cycles (at 6 ops/cycle): {min_valu_cycles:.0f}")
    print()

    # Load minimums - depends on algorithm
    # Optimal: reuse tree values when indices converge
    print("Load analysis by round type:")

    # Truly minimal loads - just the unique tree values accessed
    # After round 0: all at root (1 value)
    # After round 1: at {1,2} (2 values)
    # After round 2: at {3-6} (4 values)
    # After round 10: all wrap to root

    unique_values_by_round = {
        0: 1,  # all start at 0
        1: 2,  # {1,2}
        2: 4,  # {3,4,5,6}
        3: 8,
        4: 16,
        5: 32,
        6: 63,  # some collision
        7: 108,
        8: 159,
        9: 191,
        10: 224,
        11: 1,  # wrap
        12: 2,
        13: 4,
        14: 8,
        15: 16,
    }

    min_loads = sum(unique_values_by_round.values())
    print(f"Truly minimal loads (unique values only): {min_loads}")
    print(f"Min load cycles: {min_loads / 2:.0f}")
    print()

    # But we also need to load hash constants, input values, etc.
    init_loads = 32 + 32 + 12  # indices + values + hash consts
    store_ops = 64  # writeback at end
    overhead_cycles = init_loads / 2 + store_ops / 2

    print(f"Init/store overhead: ~{overhead_cycles:.0f} cycles")
    print()

    absolute_min = max(min_valu_cycles, min_loads / 2) + overhead_cycles
    print(f"ABSOLUTE MINIMUM (no overhead): {absolute_min:.0f} cycles")
    print()

    # What if we could achieve 100% VALU utilization?
    print("=" * 70)
    print("WHAT IF SCENARIOS")
    print("=" * 70)

    # Scenario 1: 100% VALU utilization
    print("\n1. Perfect VALU utilization (100%):")
    perfect_valu_cycles = total_valu_ops / 6 + overhead_cycles
    print(f"   Cycles: {perfect_valu_cycles:.0f}")
    print(f"   Achievable? Unlikely - data dependencies prevent perfect packing")

    # Scenario 2: Speculative for all predictable rounds
    print("\n2. Speculative execution for predictable rounds:")
    spec_rounds = [1, 2, 12, 13, 14, 15]
    loads_saved = sum((n_vectors * 8 - unique_values_by_round[r]) for r in spec_rounds)
    valu_overhead = (
        sum(n_vectors * 3 for r in spec_rounds if unique_values_by_round[r] <= 2)
        + sum(n_vectors * 6 for r in spec_rounds if 2 < unique_values_by_round[r] <= 4)
        + sum(n_vectors * 9 for r in spec_rounds if 4 < unique_values_by_round[r] <= 8)
        + sum(
            n_vectors * 12 for r in spec_rounds if 8 < unique_values_by_round[r] <= 16
        )
    )
    net_savings = loads_saved / 2 - valu_overhead / 6
    spec_cycles = current_cycles - net_savings
    print(f"   Loads saved: {loads_saved}")
    print(f"   VALU overhead: {valu_overhead}")
    print(f"   Net savings: {net_savings:.0f} cycles")
    print(f"   Result: {spec_cycles:.0f} cycles")

    # Scenario 3: Process more vectors per bundle
    print("\n3. Process 6 vectors per batch (double):")
    print("   - Would halve number of triples to process")
    print("   - But can't fit 6×2=12 ops in 6 VALU slots for hash stage1")
    print("   - NOT feasible with current hash structure")

    # Scenario 4: Reduce hash ops
    print("\n4. Reduce hash operations:")
    print("   - Current: 6 stages × 3 ops = 18 ops per vector")
    print("   - If we could do 2 stages: 6 ops per vector")
    print("   - But we CAN'T change the hash function")

    # Scenario 5: Cross-round pipelining
    print("\n5. Cross-round pipelining:")
    print("   - Process round N hash while loading round N+1 values")
    print("   - Could hide load latency but complex to implement")
    print("   - Potential: merge hash compute with loads = better utilization")

    # What would be needed
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print()
    print("To get from 3946 to <1700 cycles requires >2.3x speedup")
    print()
    print("Current VALU utilization: 58.2%")
    print("If we achieved 100% VALU util: ~2300 cycles (still >1700)")
    print()
    print("This means achieving <1700 cycles requires EITHER:")
    print("  1. Reducing total VALU ops (algorithm change - not allowed)")
    print("  2. Processing multiple rounds in parallel somehow")
    print("  3. Having load and VALU operations perfectly overlapped")
    print()
    print(
        "The key insight: normal rounds are LOAD-BOUND (8 gathers × 32 vec = 256 loads)"
    )
    print("Load bound: 256 loads / 2 per cycle = 128 cycles per normal round MINIMUM")
    print("14 normal rounds × 128 = 1792 cycles just for loads")
    print()
    print("To achieve <1700, we MUST reduce loads dramatically.")
    print("Speculative execution for predictable rounds is the primary lever.")


if __name__ == "__main__":
    main()
