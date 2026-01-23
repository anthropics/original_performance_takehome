#!/usr/bin/env python3
"""Per-cycle utilization profiler for VLIW kernels.

Shows exactly where cycles are spent and identifies underutilization.

Usage:
    python scripts/profiler.py
    python scripts/profiler.py --phase init      # First 50 cycles
    python scripts/profiler.py --phase drain     # Last 50 cycles
    python scripts/profiler.py --phase steady    # Middle section
    python scripts/profiler.py --histogram       # Utilization histogram
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SLOT_LIMITS
from perf_takehome import KernelBuilder


def get_kernel_instructions():
    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)
    return kb.instrs


def analyze_utilization(instrs):
    """Compute per-engine utilization stats."""
    engines = ["valu", "alu", "load", "store", "flow"]

    totals = {e: 0 for e in engines}
    per_cycle = {e: [] for e in engines}

    for instr in instrs:
        for e in engines:
            count = len(instr.get(e, []))
            totals[e] += count
            per_cycle[e].append(count)

    return totals, per_cycle


def print_phase(instrs, start, end, label):
    """Print detailed breakdown of a cycle range."""
    print(f"\n{'=' * 60}")
    print(f"{label} (cycles {start}-{end})")
    print("=" * 60)

    for i, instr in enumerate(instrs[start:end]):
        cycle = start + i
        parts = []
        for e in ["valu", "alu", "load", "store", "flow"]:
            count = len(instr.get(e, []))
            if count > 0:
                parts.append(f"{e}={count}")
        print(f"  {cycle:4d}: {', '.join(parts) if parts else '(empty)'}")


def print_histogram(per_cycle, engine):
    """Print utilization histogram for an engine."""
    limit = SLOT_LIMITS.get(engine, 1)
    dist = Counter(per_cycle[engine])
    total = len(per_cycle[engine])

    print(f"\n{engine.upper()} Utilization (max {limit}/cycle):")
    print("-" * 40)

    for util in range(limit + 1):
        count = dist.get(util, 0)
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {util:2d}: {count:5d} ({pct:5.1f}%) {bar}")

    avg = sum(per_cycle[engine]) / total
    print(f"\n  Average: {avg:.2f}/{limit} ({avg / limit * 100:.1f}%)")


def print_summary(instrs, totals, per_cycle):
    """Print overall summary."""
    total_cycles = len(instrs)

    print("\n" + "=" * 60)
    print("UTILIZATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal cycles: {total_cycles}")
    print(f"\nPer-engine breakdown:")
    print("-" * 40)

    for e in ["valu", "alu", "load", "store", "flow"]:
        limit = SLOT_LIMITS.get(e, 1)
        total_ops = totals[e]
        max_possible = total_cycles * limit
        util = total_ops / max_possible * 100
        avg = total_ops / total_cycles

        theoretical_min = (total_ops + limit - 1) // limit

        print(f"  {e:5s}: {total_ops:6d} ops, {avg:.2f}/{limit} avg ({util:.1f}% util)")
        print(f"         theoretical min: {theoretical_min} cycles")

    # Identify bottleneck
    print("\n" + "-" * 40)
    bottleneck_cycles = {}
    for e in ["valu", "alu", "load", "store"]:
        limit = SLOT_LIMITS[e]
        bottleneck_cycles[e] = (totals[e] + limit - 1) // limit

    bottleneck = max(bottleneck_cycles, key=bottleneck_cycles.get)
    print(
        f"BOTTLENECK: {bottleneck.upper()} ({bottleneck_cycles[bottleneck]} cycles minimum)"
    )
    print(f"Actual cycles: {total_cycles}")
    print(f"Efficiency: {bottleneck_cycles[bottleneck] / total_cycles * 100:.1f}%")


def find_bubbles(instrs, threshold=3):
    """Find cycles with low utilization (scheduling bubbles)."""
    bubbles = []

    for i, instr in enumerate(instrs):
        valu = len(instr.get("valu", []))
        if valu <= threshold:
            bubbles.append((i, valu, dict(instr)))

    return bubbles


def print_bubbles(bubbles, limit=20):
    """Print low-utilization cycles."""
    print(f"\n{'=' * 60}")
    print(f"SCHEDULING BUBBLES (VALU <= 3)")
    print("=" * 60)

    if not bubbles:
        print("  No bubbles found!")
        return

    print(f"\nFound {len(bubbles)} bubbles. Showing first {min(limit, len(bubbles))}:")
    print("-" * 40)

    for i, (cycle, valu, instr) in enumerate(bubbles[:limit]):
        parts = [f"{e}={len(ops)}" for e, ops in instr.items() if ops]
        print(f"  {cycle:4d}: VALU={valu} | {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="VLIW kernel profiler")
    parser.add_argument(
        "--phase",
        choices=["init", "drain", "steady", "all"],
        default="all",
        help="Which phase to analyze",
    )
    parser.add_argument(
        "--histogram", action="store_true", help="Show utilization histograms"
    )
    parser.add_argument(
        "--bubbles", action="store_true", help="Find scheduling bubbles"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=50,
        help="Number of cycles to show for phase analysis",
    )
    args = parser.parse_args()

    instrs = get_kernel_instructions()
    totals, per_cycle = analyze_utilization(instrs)

    print_summary(instrs, totals, per_cycle)

    if args.histogram:
        for e in ["valu", "alu", "load"]:
            print_histogram(per_cycle, e)

    if args.bubbles:
        bubbles = find_bubbles(instrs)
        print_bubbles(bubbles)

    n = args.cycles
    if args.phase == "init" or args.phase == "all":
        print_phase(instrs, 0, min(n, len(instrs)), "INIT PHASE")

    if args.phase == "drain" or args.phase == "all":
        print_phase(instrs, max(0, len(instrs) - n), len(instrs), "DRAIN PHASE")

    if args.phase == "steady":
        mid = len(instrs) // 2
        print_phase(instrs, mid - n // 2, mid + n // 2, "STEADY STATE")


if __name__ == "__main__":
    main()
