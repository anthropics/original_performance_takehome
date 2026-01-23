#!/usr/bin/env python3
"""
Kernel Analysis Script

Analyzes the generated kernel to identify optimization opportunities:
1. Bundle utilization (how well we pack operations)
2. Slot waste (idle slots per engine)
3. Cycle breakdown by phase
4. Dependency chains
"""

import sys

sys.path.insert(0, ".")

from perf_takehome import KernelBuilder
from problem import VLEN, SLOT_LIMITS


def analyze_bundle_utilization(instrs):
    """Analyze how well bundles are packed."""
    stats = {
        "total_bundles": len(instrs),
        "single_engine": 0,
        "multi_engine": 0,
        "by_engine": {},
        "valu_ops_histogram": {},  # count of bundles with N valu ops
        "load_ops_histogram": {},
    }

    for engine in SLOT_LIMITS:
        stats["by_engine"][engine] = {"bundles": 0, "ops": 0, "max_ops": 0}

    for instr in instrs:
        n_engines = len(instr)
        if n_engines == 1:
            stats["single_engine"] += 1
        else:
            stats["multi_engine"] += 1

        for engine, ops in instr.items():
            if engine in stats["by_engine"]:
                stats["by_engine"][engine]["bundles"] += 1
                stats["by_engine"][engine]["ops"] += len(ops)
                stats["by_engine"][engine]["max_ops"] = max(
                    stats["by_engine"][engine]["max_ops"], len(ops)
                )

        # Histograms
        if "valu" in instr:
            n = len(instr["valu"])
            stats["valu_ops_histogram"][n] = stats["valu_ops_histogram"].get(n, 0) + 1

        if "load" in instr:
            n = len(instr["load"])
            stats["load_ops_histogram"][n] = stats["load_ops_histogram"].get(n, 0) + 1

    return stats


def analyze_slot_waste(instrs):
    """Calculate wasted slot capacity."""
    waste = {
        "valu": {"available": 0, "used": 0},
        "load": {"available": 0, "used": 0},
        "store": {"available": 0, "used": 0},
        "alu": {"available": 0, "used": 0},
    }

    for instr in instrs:
        # Every bundle could potentially use all slots
        for engine in waste:
            if engine in SLOT_LIMITS:
                waste[engine]["available"] += SLOT_LIMITS[engine]

        for engine, ops in instr.items():
            if engine in waste:
                waste[engine]["used"] += len(ops)

    for engine in waste:
        waste[engine]["wasted"] = waste[engine]["available"] - waste[engine]["used"]
        if waste[engine]["available"] > 0:
            waste[engine]["utilization"] = (
                waste[engine]["used"] / waste[engine]["available"]
            )
        else:
            waste[engine]["utilization"] = 0

    return waste


def analyze_consecutive_patterns(instrs):
    """Find consecutive bundles that could potentially be merged."""
    patterns = {
        "consecutive_valu_only": 0,
        "consecutive_load_only": 0,
        "valu_without_load": 0,  # VALU bundles with no load that follow another
        "max_valu_only_run": 0,
    }

    current_valu_run = 0
    prev_was_valu_only = False

    for instr in instrs:
        is_valu_only = list(instr.keys()) == ["valu"]
        is_load_only = list(instr.keys()) == ["load"]
        has_valu = "valu" in instr
        has_load = "load" in instr

        if is_valu_only:
            current_valu_run += 1
            patterns["max_valu_only_run"] = max(
                patterns["max_valu_only_run"], current_valu_run
            )
            if prev_was_valu_only:
                patterns["consecutive_valu_only"] += 1
        else:
            current_valu_run = 0

        if is_load_only and prev_was_valu_only:
            patterns["consecutive_load_only"] += 1

        if has_valu and not has_load:
            patterns["valu_without_load"] += 1

        prev_was_valu_only = is_valu_only

    return patterns


def find_mergeable_bundles(instrs):
    """Find pairs of consecutive bundles that could be merged."""
    mergeable = []

    for i in range(len(instrs) - 1):
        curr = instrs[i]
        next_b = instrs[i + 1]

        # Check if they use different engines or have room
        can_merge = True
        merged = {}

        for engine in set(curr.keys()) | set(next_b.keys()):
            if engine == "debug":
                continue
            curr_ops = curr.get(engine, [])
            next_ops = next_b.get(engine, [])
            total = len(curr_ops) + len(next_ops)

            if engine in SLOT_LIMITS and total > SLOT_LIMITS[engine]:
                can_merge = False
                break

            merged[engine] = curr_ops + next_ops

        if can_merge and merged:
            mergeable.append((i, merged))

    return mergeable[:20]  # Return first 20 examples


def analyze_phase_breakdown(instrs, n_vectors=32, n_rounds=16):
    """Estimate cycles spent in each phase."""
    # This is approximate based on code structure
    phases = {
        "setup": 0,
        "load_initial": 0,
        "broadcast_rounds": 0,
        "gather_rounds": 0,
        "store_final": 0,
    }

    # Count bundles (each bundle = 1 cycle)
    # Setup: first ~50 bundles (hash constants, vector constants)
    # Load initial: next ~192 bundles (32 vectors × 3 bundles each × 2 for idx+val)
    # Rounds: bulk of the work
    # Store final: last ~192 bundles

    phases["total"] = len(instrs)

    # Rough estimates based on code structure
    phases["setup"] = 50
    phases["load_initial"] = 192
    phases["store_final"] = 192
    phases["rounds"] = (
        phases["total"]
        - phases["setup"]
        - phases["load_initial"]
        - phases["store_final"]
    )

    return phases


def main():
    print("=" * 60)
    print("KERNEL ANALYSIS REPORT")
    print("=" * 60)

    kb = KernelBuilder()
    kb.build_kernel(10, 1023, 256, 16)
    instrs = kb.instrs

    print(f"\nTotal bundles (cycles): {len(instrs)}")
    print(f"Scratch space used: {kb.scratch_ptr} / 1536 words")

    # Bundle utilization
    print("\n" + "-" * 40)
    print("BUNDLE UTILIZATION")
    print("-" * 40)

    util = analyze_bundle_utilization(instrs)
    print(
        f"Single-engine bundles: {util['single_engine']} ({100 * util['single_engine'] / util['total_bundles']:.1f}%)"
    )
    print(
        f"Multi-engine bundles: {util['multi_engine']} ({100 * util['multi_engine'] / util['total_bundles']:.1f}%)"
    )

    print("\nVALU ops per bundle histogram:")
    for n in sorted(util["valu_ops_histogram"].keys()):
        count = util["valu_ops_histogram"][n]
        bar = "█" * (count // 50)
        print(f"  {n} ops: {count:4d} {bar}")

    print("\nLoad ops per bundle histogram:")
    for n in sorted(util["load_ops_histogram"].keys()):
        count = util["load_ops_histogram"][n]
        bar = "█" * (count // 20)
        print(f"  {n} ops: {count:4d} {bar}")

    # Slot waste
    print("\n" + "-" * 40)
    print("SLOT WASTE ANALYSIS")
    print("-" * 40)

    waste = analyze_slot_waste(instrs)
    for engine in ["valu", "load", "alu", "store"]:
        w = waste[engine]
        print(
            f"{engine.upper():5s}: {w['used']:5d} used / {w['available']:6d} available = {100 * w['utilization']:.1f}% utilization ({w['wasted']} wasted)"
        )

    # Patterns
    print("\n" + "-" * 40)
    print("CONSECUTIVE PATTERNS")
    print("-" * 40)

    patterns = analyze_consecutive_patterns(instrs)
    print(f"Consecutive VALU-only bundles: {patterns['consecutive_valu_only']}")
    print(f"Max VALU-only run length: {patterns['max_valu_only_run']}")
    print(f"VALU bundles without loads: {patterns['valu_without_load']}")

    # Mergeable bundles
    print("\n" + "-" * 40)
    print("MERGEABLE BUNDLE EXAMPLES")
    print("-" * 40)

    mergeable = find_mergeable_bundles(instrs)
    print(
        f"Found {len(mergeable)} potentially mergeable consecutive pairs (showing first 10):"
    )
    for i, (idx, merged) in enumerate(mergeable[:10]):
        engines = list(merged.keys())
        ops = {e: len(merged[e]) for e in engines}
        print(f"  Bundles {idx},{idx + 1}: {ops}")

    # Phase breakdown
    print("\n" + "-" * 40)
    print("PHASE BREAKDOWN (estimated)")
    print("-" * 40)

    phases = analyze_phase_breakdown(instrs)
    for phase, cycles in phases.items():
        if phase != "total":
            print(
                f"{phase:15s}: {cycles:5d} cycles ({100 * cycles / phases['total']:.1f}%)"
            )

    # Recommendations
    print("\n" + "=" * 60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 60)

    if waste["load"]["wasted"] > 1000:
        print(f"⚠ HIGH LOAD SLOT WASTE: {waste['load']['wasted']} idle load slots")
        print("  → Consider adding prefetch/speculative loads during VALU-only cycles")

    if patterns["consecutive_valu_only"] > 100:
        print(f"⚠ CONSECUTIVE VALU-ONLY: {patterns['consecutive_valu_only']} pairs")
        print("  → These cycles have completely idle Load slots")

    if util["valu_ops_histogram"].get(1, 0) > 500:
        print(
            f"⚠ UNDER-PACKED VALU: {util['valu_ops_histogram'].get(1, 0)} bundles with only 1 VALU op"
        )
        print("  → Consider merging operations or restructuring loops")

    valu_under_4 = sum(util["valu_ops_histogram"].get(n, 0) for n in range(1, 4))
    if valu_under_4 > 1000:
        print(f"⚠ VALU CAPACITY: {valu_under_4} bundles using <4 of 6 VALU slots")


if __name__ == "__main__":
    main()
