"""Detailed cycle analysis to find remaining optimizations"""

from perf_takehome import KernelBuilder
from problem import HASH_STAGES, VLEN

def analyze_cycle_breakdown():
    """Analyze where cycles are being spent"""
    kb = KernelBuilder()
    kb.build_kernel(forest_height=10, n_nodes=1024, batch_size=256, rounds=16)

    # Count non-debug bundles by phase
    phase_cycles = {
        "init": 0,
        "load": 0,
        "indirect_addr": 0,
        "indirect_load": 0,
        "xor": 0,
        "hash": 0,
        "index_calc": 0,
        "bounds": 0,
        "store": 0,
        "other": 0,
    }

    non_debug_count = 0
    for instr in kb.instrs:
        # Count only non-debug bundles
        has_non_debug = any(engine != "debug" for engine in instr.keys())
        if not has_non_debug:
            continue

        non_debug_count += 1

        # Categorize by dominant operation
        if "load" in instr:
            if any("vload" in str(op) for op in instr["load"]):
                phase_cycles["load"] += 1
            else:
                phase_cycles["indirect_load"] += 1
        elif "store" in instr:
            phase_cycles["store"] += 1
        elif "valu" in instr:
            # Check if hash, xor, or index calc
            ops = [op[0] if isinstance(op, tuple) else "" for op in instr["valu"]]
            if "^" in ops and len(instr["valu"]) <= 6:
                phase_cycles["xor"] += 1
            elif "%" in ops or "*" in ops:
                phase_cycles["index_calc"] += 1
            elif "<" in ops:
                phase_cycles["bounds"] += 1
            else:
                phase_cycles["hash"] += 1
        elif "alu" in instr:
            phase_cycles["indirect_addr"] += 1
        elif "flow" in instr:
            phase_cycles["other"] += 1
        else:
            phase_cycles["init"] += 1

    print(f"Total non-debug cycles: {non_debug_count}")
    print(f"\nCycle breakdown by phase:")
    for phase, count in sorted(phase_cycles.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = 100 * count / non_debug_count
            print(f"  {phase:20s}: {count:5d} cycles ({pct:5.1f}%)")

    # Calculate theoretical minimums
    print(f"\nTheoretical minimums:")
    batch_count = 32 / 6  # groups / parallel
    round_count = 16

    # Indirect loads: 32 groups * 16 rounds * 8 loads per group / 2 slots
    indirect_loads_theory = (32 * 16 * 8) / 2
    print(f"  Indirect loads: {indirect_loads_theory:.0f} cycles (actual: {phase_cycles['indirect_load']})")

    # Hash: 6 stages * 2 bundles per stage * 16 rounds * ~5.33 batches
    hash_theory = 6 * 2 * 16 * (32/6)
    print(f"  Hash operations: {hash_theory:.0f} cycles (actual: {phase_cycles['hash']})")

    print(f"\n  Current cycles: {non_debug_count}")
    print(f"  Theoretical minimum: {indirect_loads_theory + hash_theory:.0f}")
    print(f"  Overhead: {non_debug_count - (indirect_loads_theory + hash_theory):.0f} cycles")

if __name__ == "__main__":
    analyze_cycle_breakdown()
