#!/usr/bin/env python3
"""Detect whether kernel is VALU-bound, memory-bound, or something else.

Provides actionable recommendations based on the bottleneck.

Usage:
    python scripts/bottleneck_detector.py
"""

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


def count_ops(instrs):
    """Count total operations per engine."""
    totals = Counter()
    for instr in instrs:
        for engine, ops in instr.items():
            totals[engine] += len(ops)
    return totals


def compute_theoretical_min(totals):
    """Compute theoretical minimum cycles for each engine."""
    mins = {}
    for engine, count in totals.items():
        limit = SLOT_LIMITS.get(engine, 1)
        mins[engine] = (count + limit - 1) // limit
    return mins


def analyze_bottleneck(instrs):
    """Determine the bottleneck and provide recommendations."""
    totals = count_ops(instrs)
    mins = compute_theoretical_min(totals)
    actual = len(instrs)

    # Find the bottleneck
    bottleneck_engine = max(mins, key=mins.get)
    bottleneck_cycles = mins[bottleneck_engine]

    # Calculate efficiency
    efficiency = bottleneck_cycles / actual * 100

    # Compute actual utilization
    utilization = {}
    for engine in ["valu", "alu", "load", "store", "flow"]:
        limit = SLOT_LIMITS.get(engine, 1)
        max_possible = actual * limit
        utilization[engine] = (
            totals.get(engine, 0) / max_possible * 100 if max_possible > 0 else 0
        )

    return {
        "totals": totals,
        "theoretical_mins": mins,
        "actual_cycles": actual,
        "bottleneck": bottleneck_engine,
        "bottleneck_cycles": bottleneck_cycles,
        "efficiency": efficiency,
        "utilization": utilization,
    }


def print_report(analysis):
    """Print bottleneck analysis report."""
    print("=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)

    print(f"\nActual cycles: {analysis['actual_cycles']}")
    print(f"\nOperation counts:")
    print("-" * 40)

    for engine in ["valu", "alu", "load", "store", "flow"]:
        count = analysis["totals"].get(engine, 0)
        limit = SLOT_LIMITS.get(engine, 1)
        min_cycles = analysis["theoretical_mins"].get(engine, 0)
        util = analysis["utilization"].get(engine, 0)

        marker = " <-- BOTTLENECK" if engine == analysis["bottleneck"] else ""
        print(
            f"  {engine:5s}: {count:6d} ops / {limit} per cycle = {min_cycles:5d} min cycles ({util:5.1f}% util){marker}"
        )

    print(f"\n" + "=" * 70)
    print(f"DIAGNOSIS: {analysis['bottleneck'].upper()}-BOUND")
    print("=" * 70)

    print(f"\nTheoretical minimum: {analysis['bottleneck_cycles']} cycles")
    print(f"Actual: {analysis['actual_cycles']} cycles")
    print(f"Efficiency: {analysis['efficiency']:.1f}%")
    print(f"Gap: {analysis['actual_cycles'] - analysis['bottleneck_cycles']} cycles")

    # Recommendations based on bottleneck
    print(f"\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)

    bottleneck = analysis["bottleneck"]

    if bottleneck == "valu":
        print(
            """
VALU-BOUND kernel. To improve:

1. REDUCE VALU OPS
   - Use multiply_add fusion: val + const + (val << shift) -> multiply_add(val, 1+(1<<shift), const)
   - Fuse hash stages where pattern allows (op1=='+' and op2=='+' and op3=='<<')
   - Check for redundant vector operations

2. IMPROVE VALU PACKING
   - Current utilization: {:.1f}%
   - Examine cycles with <6 VALU - why can't more ops be packed?
   - Data dependencies often force serialization
   - Consider reordering operations to break dependency chains

3. ALGORITHMIC CHANGES (if allowed)
   - Reduce total work (fewer rounds, smaller batches)
   - Approximate hash computation
""".format(analysis["utilization"]["valu"])
        )

    elif bottleneck == "load":
        print(
            """
MEMORY-BOUND kernel. To improve:

1. REDUCE LOADS
   - Use vselect + preloaded constants instead of gathers for predictable indices
   - Cache frequently accessed values in scratch
   - Batch loads to maximize load slot utilization

2. HIDE LOAD LATENCY  
   - Software pipelining: load round N+1 while computing round N
   - Prefetch during hash computation (idle load slots)
   - Current load utilization: {:.1f}%

3. ARCHITECTURAL CHANGES
   - Consider speculative execution for predictable tree levels
   - Trade VALU ops for reduced memory access
""".format(analysis["utilization"]["load"])
        )

    elif bottleneck == "store":
        print("""
STORE-BOUND kernel (unusual). To improve:

1. REDUCE STORES
   - Only store final results (not intermediate)
   - Check for redundant stores

2. BATCH STORES
   - Accumulate results, store at end
   - Use vstore instead of scalar stores
""")

    elif bottleneck == "flow":
        print(
            """
FLOW-BOUND kernel. To improve:

1. ELIMINATE vselect OPERATIONS
   - Replace vselect with arithmetic:
     - vselect(cond, a, b) -> a*cond + b*(1-cond) [if values are numeric]
     - vselect(cond, 1, 2) -> 1 + (1-cond) [simpler]
   
2. ELIMINATE CONTROL FLOW
   - Unroll loops fully
   - Remove conditional jumps
   - Current flow utilization: {:.1f}%
""".format(analysis["utilization"]["flow"])
        )

    # Universal recommendations
    print("""
UNIVERSAL OPTIMIZATIONS:
   - Run profiler.py to find scheduling bubbles
   - Run visualize_schedule.py --gaps to find contiguous low-utilization regions
   - Check init and drain phases for optimization opportunities
""")


def main():
    instrs = get_kernel_instructions()
    analysis = analyze_bottleneck(instrs)
    print_report(analysis)


if __name__ == "__main__":
    main()
