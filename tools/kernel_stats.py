from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Iterable

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from perf_takehome import KernelBuilder
from problem import SCRATCH_SIZE, SLOT_LIMITS, Input, Tree


def build_kernel(forest_height: int, rounds: int, batch_size: int, seed: int):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    return kb


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_counter(counter: Counter[str], limit: int) -> str:
    items = counter.most_common(limit)
    return ", ".join(f"{name}={count}" for name, count in items)


def analyze_program(program: list[dict], max_ops: int):
    total_instrs = len(program)
    effective_cycles = sum(1 for instr in program if any(k != "debug" for k in instr))
    debug_only = sum(1 for instr in program if all(k == "debug" for k in instr))

    engine_slot_counts: Counter[str] = Counter()
    engine_bundle_counts: Counter[str] = Counter()
    op_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bundle_slots: list[int] = []

    for instr in program:
        total_slots = 0
        for engine, slots in instr.items():
            slot_count = len(slots)
            engine_slot_counts[engine] += slot_count
            if slot_count:
                engine_bundle_counts[engine] += 1
            for slot in slots:
                op_counts[engine][str(slot[0])] += 1
            if engine != "debug":
                total_slots += slot_count
        if total_slots:
            bundle_slots.append(total_slots)

    return {
        "total_instrs": total_instrs,
        "effective_cycles": effective_cycles,
        "debug_only": debug_only,
        "engine_slot_counts": engine_slot_counts,
        "engine_bundle_counts": engine_bundle_counts,
        "op_counts": op_counts,
        "bundle_slots": bundle_slots,
        "max_ops": max_ops,
    }


def print_summary(kb: KernelBuilder, stats: dict):
    total_instrs = stats["total_instrs"]
    effective_cycles = stats["effective_cycles"] or 1
    debug_only = stats["debug_only"]
    engine_slot_counts = stats["engine_slot_counts"]
    engine_bundle_counts = stats["engine_bundle_counts"]
    op_counts = stats["op_counts"]
    bundle_slots = stats["bundle_slots"]
    max_ops = stats["max_ops"]

    avg_bundle_slots = sum(bundle_slots) / len(bundle_slots) if bundle_slots else 0.0
    max_bundle_slots = max(bundle_slots) if bundle_slots else 0

    print("Kernel program stats")
    print(f"  Instruction bundles: {total_instrs}")
    print(f"  Effective cycles:    {effective_cycles}")
    print(f"  Debug-only bundles:  {debug_only}")
    print(f"  Avg slots/bundle:    {avg_bundle_slots:.2f}")
    print(f"  Max slots/bundle:    {max_bundle_slots}")
    print(
        f"  Scratch used:        {kb.scratch_ptr}/{SCRATCH_SIZE}"
        f" ({_format_pct(kb.scratch_ptr / SCRATCH_SIZE)})"
    )
    print(f"  Const entries:       {len(kb.const_map)}")
    print("\nEngine utilization")

    for engine, limit in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        total_slots = engine_slot_counts.get(engine, 0)
        avg_slots = total_slots / effective_cycles
        utilization = total_slots / (effective_cycles * limit)
        bundles_with_engine = engine_bundle_counts.get(engine, 0)
        bundle_pct = bundles_with_engine / total_instrs if total_instrs else 0.0
        print(
            f"  {engine:5} | slots={total_slots:7} | avg/cycle={avg_slots:5.2f}"
            f" | util={_format_pct(utilization):>7} | bundles={bundle_pct:6.2%}"
        )

    if op_counts:
        print("\nOp breakdown")
        for engine, counter in op_counts.items():
            summary = _format_counter(counter, max_ops)
            print(f"  {engine:5} | {summary}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect kernel instruction stats")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-ops", type=int, default=12)
    args = parser.parse_args(list(argv) if argv is not None else None)

    kb = build_kernel(args.forest_height, args.rounds, args.batch_size, args.seed)
    stats = analyze_program(kb.instrs, args.max_ops)
    print_summary(kb, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
