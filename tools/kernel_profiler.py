from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from perf_takehome import KernelBuilder
from problem import SLOT_LIMITS, Input, Tree

ENGINE_ORDER = [engine for engine in SLOT_LIMITS.keys() if engine != "debug"]


@dataclass(frozen=True)
class SegmentLabel:
    round: Optional[int]
    index: Optional[int]
    stage: str

    def iteration_key(self) -> Optional[tuple[int, int]]:
        if self.round is None or self.index is None:
            return None
        return (self.round, self.index)


@dataclass
class Stats:
    instr_count: int = 0
    cycles: int = 0
    debug_only: int = 0
    total_slots: int = 0
    bundle_slots: list[int] = field(default_factory=list)
    slot_counts: Counter[str] = field(default_factory=Counter)
    op_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def add_instruction(self, instr: dict) -> None:
        self.instr_count += 1
        non_debug_slots = 0
        debug_only = True
        for engine, slots in instr.items():
            if engine != "debug":
                debug_only = False
                slot_count = len(slots)
                self.slot_counts[engine] += slot_count
                self.total_slots += slot_count
                non_debug_slots += slot_count
                for slot in slots:
                    self.op_counts[engine][str(slot[0])] += 1
        if debug_only:
            self.debug_only += 1
        else:
            self.cycles += 1
            self.bundle_slots.append(non_debug_slots)

    def merge(self, other: "Stats") -> None:
        self.instr_count += other.instr_count
        self.cycles += other.cycles
        self.debug_only += other.debug_only
        self.total_slots += other.total_slots
        self.bundle_slots.extend(other.bundle_slots)
        self.slot_counts.update(other.slot_counts)
        for engine, counter in other.op_counts.items():
            self.op_counts[engine].update(counter)


@dataclass
class Segment:
    label: SegmentLabel
    start_index: int
    end_index: int
    stats: Stats = field(default_factory=Stats)


def build_kernel(forest_height: int, rounds: int, batch_size: int, seed: int):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    return kb


def is_pause(instr: dict) -> bool:
    for slot in instr.get("flow", []):
        if slot and slot[0] == "pause":
            return True
    return False


def label_from_debug(instr: dict) -> Optional[SegmentLabel]:
    for slot in instr.get("debug", []):
        if slot[0] == "compare":
            key = slot[2]
        elif slot[0] == "vcompare":
            keys = slot[2]
            key = keys[0] if keys else None
        else:
            continue

        if isinstance(key, tuple) and len(key) >= 3:
            stage = key[2]
            if stage == "hash_stage" and len(key) >= 4:
                stage = f"hash_stage_{key[3]}"
            return SegmentLabel(round=key[0], index=key[1], stage=stage)
    return None


def split_segments(program: list[dict]) -> list[Segment]:
    segments: list[Segment] = []
    current = Segment(
        label=SegmentLabel(round=None, index=None, stage="prologue"),
        start_index=0,
        end_index=-1,
    )

    for idx, instr in enumerate(program):
        if is_pause(instr):
            if current.stats.instr_count:
                current.end_index = idx - 1
                segments.append(current)
            pause_seg = Segment(
                label=SegmentLabel(round=None, index=None, stage="pause"),
                start_index=idx,
                end_index=idx,
            )
            pause_seg.stats.add_instruction(instr)
            segments.append(pause_seg)
            current = Segment(
                label=SegmentLabel(round=None, index=None, stage="unlabeled"),
                start_index=idx + 1,
                end_index=-1,
            )
            continue

        current.stats.add_instruction(instr)
        label = label_from_debug(instr)
        if label:
            current.label = label
            current.end_index = idx
            segments.append(current)
            current = Segment(
                label=SegmentLabel(round=None, index=None, stage="unlabeled"),
                start_index=idx + 1,
                end_index=-1,
            )

    if current.stats.instr_count:
        current.end_index = len(program) - 1
        if current.label.stage in {"unlabeled", "prologue"}:
            current.label = SegmentLabel(round=None, index=None, stage="epilogue")
        segments.append(current)

    return segments


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def summarize_stats(stats: Stats) -> dict:
    avg_slots = sum(stats.bundle_slots) / len(stats.bundle_slots) if stats.bundle_slots else 0.0
    max_slots = max(stats.bundle_slots) if stats.bundle_slots else 0
    engine_util = {}
    for engine in ENGINE_ORDER:
        limit = SLOT_LIMITS[engine]
        slots = stats.slot_counts.get(engine, 0)
        util = slots / (stats.cycles * limit) if stats.cycles else 0.0
        engine_util[engine] = {
            "slots": slots,
            "utilization": util,
            "avg_slots_per_cycle": slots / stats.cycles if stats.cycles else 0.0,
        }
    return {
        "instr_count": stats.instr_count,
        "cycles": stats.cycles,
        "debug_only": stats.debug_only,
        "total_slots": stats.total_slots,
        "avg_slots_per_bundle": avg_slots,
        "max_slots_per_bundle": max_slots,
        "engine": engine_util,
    }


def print_overall(stats: Stats) -> None:
    summary = summarize_stats(stats)
    print("Kernel profiling summary")
    print(f"  Instruction bundles: {summary['instr_count']}")
    print(f"  Non-debug cycles:    {summary['cycles']}")
    print(f"  Debug-only bundles:  {summary['debug_only']}")
    print(f"  Avg slots/bundle:    {summary['avg_slots_per_bundle']:.2f}")
    print(f"  Max slots/bundle:    {summary['max_slots_per_bundle']}")
    print("  Engine utilization:")
    for engine in ENGINE_ORDER:
        engine_stats = summary["engine"][engine]
        print(
            f"    {engine:5} | slots={engine_stats['slots']:7}"
            f" | avg/cycle={engine_stats['avg_slots_per_cycle']:.2f}"
            f" | util={format_pct(engine_stats['utilization']):>7}"
        )


def print_stage_summary(stage_stats: dict[str, Stats], limit: int) -> None:
    ranked = sorted(stage_stats.items(), key=lambda item: item[1].cycles, reverse=True)
    print("\nStage hotspots")
    header = "stage             cycles  avg_slots  " + "  ".join(
        f"{engine[:4]}%" for engine in ENGINE_ORDER
    )
    print(header)
    print("-" * len(header))
    for stage, stats in ranked[:limit]:
        avg_slots = sum(stats.bundle_slots) / len(stats.bundle_slots) if stats.bundle_slots else 0.0
        util_cols = []
        for engine in ENGINE_ORDER:
            limit_slots = SLOT_LIMITS[engine]
            slots = stats.slot_counts.get(engine, 0)
            util = slots / (stats.cycles * limit_slots) if stats.cycles else 0.0
            util_cols.append(f"{util*100:5.1f}")
        print(f"{stage:17} {stats.cycles:7}  {avg_slots:8.2f}  " + "  ".join(util_cols))


def print_iteration_summary(iter_stats: dict[tuple[int, int], Stats], top_n: int) -> None:
    ranked = sorted(iter_stats.items(), key=lambda item: item[1].cycles, reverse=True)
    if not ranked:
        return
    print("\nTop iterations by cycles")
    print("round idx  cycles  avg_slots")
    print("----------------------------")
    for (round_i, idx), stats in ranked[:top_n]:
        avg_slots = sum(stats.bundle_slots) / len(stats.bundle_slots) if stats.bundle_slots else 0.0
        print(f"{round_i:5} {idx:3}  {stats.cycles:6}  {avg_slots:8.2f}")


def print_iteration_timeline(segments: list[Segment], round_i: int, idx_i: int) -> None:
    rows = [seg for seg in segments if seg.label.round == round_i and seg.label.index == idx_i]
    if not rows:
        print(f"No segments found for round={round_i} idx={idx_i}")
        return
    print(f"\nIteration timeline round={round_i} idx={idx_i}")
    print("stage             cycles  avg_slots")
    print("-------------------------------")
    for seg in rows:
        avg_slots = (
            sum(seg.stats.bundle_slots) / len(seg.stats.bundle_slots)
            if seg.stats.bundle_slots
            else 0.0
        )
        print(f"{seg.label.stage:17} {seg.stats.cycles:7}  {avg_slots:8.2f}")


def build_profile(segments: list[Segment]):
    overall = Stats()
    stage_stats: dict[str, Stats] = defaultdict(Stats)
    iter_stats: dict[tuple[int, int], Stats] = defaultdict(Stats)

    for seg in segments:
        overall.merge(seg.stats)
        stage_stats[seg.label.stage].merge(seg.stats)
        iteration_key = seg.label.iteration_key()
        if iteration_key is not None:
            iter_stats[iteration_key].merge(seg.stats)

    return overall, stage_stats, iter_stats


def encode_stats(stats: Stats) -> dict:
    summary = summarize_stats(stats)
    summary["op_counts"] = {
        engine: dict(counter) for engine, counter in stats.op_counts.items()
    }
    return summary


def segments_to_json(segments: list[Segment]) -> list[dict]:
    result = []
    for seg in segments:
        result.append(
            {
                "label": {
                    "round": seg.label.round,
                    "index": seg.label.index,
                    "stage": seg.label.stage,
                },
                "start_index": seg.start_index,
                "end_index": seg.end_index,
                "stats": encode_stats(seg.stats),
            }
        )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile kernel hotspots and utilization")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--top-stages", type=int, default=12)
    parser.add_argument("--top-iterations", type=int, default=10)
    parser.add_argument("--iteration", type=str, default="")
    parser.add_argument("--json", dest="json_path", type=str, default="")
    parser.add_argument("--include-segments", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    kb = build_kernel(args.forest_height, args.rounds, args.batch_size, args.seed)
    segments = split_segments(kb.instrs)
    overall, stage_stats, iter_stats = build_profile(segments)

    print_overall(overall)
    print_stage_summary(stage_stats, args.top_stages)
    print_iteration_summary(iter_stats, args.top_iterations)

    if args.iteration:
        round_str, idx_str = args.iteration.split(",", maxsplit=1)
        print_iteration_timeline(segments, int(round_str), int(idx_str))

    if args.json_path:
        payload = {
            "summary": encode_stats(overall),
            "stages": {stage: encode_stats(stats) for stage, stats in stage_stats.items()},
            "iterations": {
                f"{key[0]},{key[1]}": encode_stats(stats) for key, stats in iter_stats.items()
            },
        }
        if args.include_segments:
            payload["segments"] = segments_to_json(segments)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
