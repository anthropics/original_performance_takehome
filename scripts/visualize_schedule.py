#!/usr/bin/env python3
"""Visualize VLIW schedule as ASCII timeline showing gaps and utilization.

Usage:
    python scripts/visualize_schedule.py
    python scripts/visualize_schedule.py --start 0 --end 100
    python scripts/visualize_schedule.py --compact
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SLOT_LIMITS
from perf_takehome import KernelBuilder


def get_kernel_instructions():
    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)
    return kb.instrs


def render_bar(count, limit, width=10):
    """Render a utilization bar."""
    filled = int(count / limit * width)
    empty = width - filled

    if count == 0:
        return "." * width
    elif count == limit:
        return "#" * width
    else:
        return "#" * filled + "-" * empty


def visualize_range(instrs, start, end, compact=False):
    """Visualize a range of cycles."""
    engines = ["valu", "alu", "load", "store", "flow"]
    limits = {e: SLOT_LIMITS.get(e, 1) for e in engines}

    if not compact:
        print(
            f"\n{'Cycle':>6} | {'VALU':^10} | {'ALU':^12} | {'LOAD':^4} | {'STORE':^5} | F"
        )
        print("-" * 60)

    for i in range(start, min(end, len(instrs))):
        instr = instrs[i]
        counts = {e: len(instr.get(e, [])) for e in engines}

        if compact:
            # Single character per engine
            valu_char = str(counts["valu"]) if counts["valu"] < 10 else "+"
            alu_char = (
                "A"
                if counts["alu"] >= 10
                else str(counts["alu"])
                if counts["alu"] > 0
                else "."
            )
            load_char = str(counts["load"]) if counts["load"] > 0 else "."
            store_char = str(counts["store"]) if counts["store"] > 0 else "."
            flow_char = "F" if counts["flow"] > 0 else "."

            print(
                f"{i:4d} {valu_char}{alu_char}{load_char}{store_char}{flow_char}",
                end="",
            )
            if (i - start + 1) % 10 == 0:
                print()
        else:
            valu_bar = render_bar(counts["valu"], 6, 10)
            alu_bar = render_bar(counts["alu"], 12, 12)
            load_bar = render_bar(counts["load"], 2, 4)
            store_bar = render_bar(counts["store"], 2, 5)
            flow_bar = "#" if counts["flow"] > 0 else "."

            print(
                f"{i:6d} | {valu_bar} | {alu_bar} | {load_bar} | {store_bar} | {flow_bar}"
            )

    if compact:
        print()


def find_gaps(instrs, valu_threshold=3):
    """Find contiguous regions of low VALU utilization."""
    gaps = []
    gap_start = None

    for i, instr in enumerate(instrs):
        valu = len(instr.get("valu", []))

        if valu <= valu_threshold:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gaps.append((gap_start, i - 1, i - gap_start))
                gap_start = None

    if gap_start is not None:
        gaps.append((gap_start, len(instrs) - 1, len(instrs) - gap_start))

    return gaps


def print_gap_summary(gaps):
    """Print summary of scheduling gaps."""
    print("\n" + "=" * 60)
    print("SCHEDULING GAPS (contiguous cycles with VALU <= 3)")
    print("=" * 60)

    if not gaps:
        print("No significant gaps found!")
        return

    # Sort by length descending
    gaps_sorted = sorted(gaps, key=lambda x: x[2], reverse=True)

    total_gap_cycles = sum(g[2] for g in gaps)
    print(f"\nTotal gap cycles: {total_gap_cycles}")
    print(f"Number of gaps: {len(gaps)}")
    print(f"\nTop 10 gaps:")
    print("-" * 40)

    for start, end, length in gaps_sorted[:10]:
        print(f"  Cycles {start:4d}-{end:4d}: {length:3d} cycles")

    # Categorize by position
    total = len(gaps_sorted)
    init_gaps = [g for g in gaps if g[0] < 50]
    drain_gaps = [g for g in gaps if g[1] > total - 50]
    mid_gaps = [g for g in gaps if g[0] >= 50 and g[1] <= total - 50]

    print(f"\nBy position:")
    print(
        f"  Init phase (first 50): {len(init_gaps)} gaps, {sum(g[2] for g in init_gaps)} cycles"
    )
    print(
        f"  Drain phase (last 50): {len(drain_gaps)} gaps, {sum(g[2] for g in drain_gaps)} cycles"
    )
    print(f"  Steady state: {len(mid_gaps)} gaps, {sum(g[2] for g in mid_gaps)} cycles")


def main():
    parser = argparse.ArgumentParser(description="Visualize VLIW schedule")
    parser.add_argument("--start", type=int, default=0, help="Start cycle")
    parser.add_argument("--end", type=int, default=50, help="End cycle")
    parser.add_argument("--compact", action="store_true", help="Compact visualization")
    parser.add_argument("--gaps", action="store_true", help="Analyze scheduling gaps")
    parser.add_argument("--all", action="store_true", help="Show all cycles (compact)")
    args = parser.parse_args()

    instrs = get_kernel_instructions()
    total = len(instrs)

    print(f"Total cycles: {total}")

    if args.gaps:
        gaps = find_gaps(instrs)
        print_gap_summary(gaps)

    if args.all:
        print("\nFull schedule (compact):")
        print("Legend: V=VALU(0-6) A=ALU(10+=A) L=Load(0-2) S=Store(0-2) F=Flow")
        print("-" * 60)
        visualize_range(instrs, 0, total, compact=True)
    else:
        visualize_range(instrs, args.start, args.end, args.compact)


if __name__ == "__main__":
    main()
