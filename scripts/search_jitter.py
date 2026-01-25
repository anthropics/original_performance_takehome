#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from pathlib import Path


def set_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def build_cycles(
    forest_height: int,
    rounds: int,
    batch_size: int,
    groups_per_batch: int,
    spacing: int,
    tail_spacing: int,
    parity_mask: int,
    pattern: list[int] | None,
    pattern_tail: list[int] | None,
) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from perf_takehome import KernelBuilder

    set_env("GROUPS_PER_BATCH", str(groups_per_batch))
    set_env("START_SPACING", str(spacing))
    set_env("START_SPACING_TAIL", str(tail_spacing))
    set_env("PARITY_MASK", str(parity_mask))
    set_env("START_PERIOD", None)
    set_env(
        "START_JITTER_PATTERN",
        ",".join(str(v) for v in pattern) if pattern else None,
    )
    set_env(
        "START_JITTER_PATTERN_TAIL",
        ",".join(str(v) for v in pattern_tail) if pattern_tail else None,
    )

    kb = KernelBuilder()
    n_nodes = 2 ** (forest_height + 1) - 1
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return len(kb.instrs)


def parse_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    return [int(part) for part in parts]


def main() -> int:
    parser = argparse.ArgumentParser(description="Search start-offset jitter patterns.")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--spacing", type=int, default=7)
    parser.add_argument("--tail-spacing", type=int, default=11)
    parser.add_argument("--parity-mask", type=int, default=70)
    parser.add_argument("--pattern-len", type=int, default=4)
    parser.add_argument("--pattern-len-tail", type=int)
    parser.add_argument("--jitter-max", type=int, default=6)
    parser.add_argument("--mode", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pattern", help="Fixed jitter pattern (comma-separated)")
    parser.add_argument("--pattern-tail", help="Fixed tail jitter pattern (comma-separated)")

    args = parser.parse_args()

    pattern = parse_list(args.pattern)
    pattern_tail = parse_list(args.pattern_tail)
    if pattern is not None:
        cycles = build_cycles(
            args.forest_height,
            args.rounds,
            args.batch_size,
            args.groups,
            args.spacing,
            args.tail_spacing,
            args.parity_mask,
            pattern,
            pattern_tail,
        )
        print(f"pattern={pattern} tail={pattern_tail} cycles={cycles}")
        return 0

    if args.seed is not None:
        random.seed(args.seed)

    base_cycles = build_cycles(
        args.forest_height,
        args.rounds,
        args.batch_size,
        args.groups,
        args.spacing,
        args.tail_spacing,
        args.parity_mask,
        None,
        None,
    )
    print(f"baseline cycles={base_cycles}")

    best = (base_cycles, None, None)
    if args.mode == "exhaustive":
        values = list(range(args.jitter_max + 1))
        tail_len = args.pattern_len_tail or args.pattern_len
        for pattern in itertools.product(values, repeat=args.pattern_len):
            if tail_len == args.pattern_len:
                patterns = [(list(pattern), None)]
            else:
                patterns = [
                    (list(pattern), list(tail))
                    for tail in itertools.product(values, repeat=tail_len)
                ]
            for pat, tail in patterns:
                cycles = build_cycles(
                    args.forest_height,
                    args.rounds,
                    args.batch_size,
                    args.groups,
                    args.spacing,
                    args.tail_spacing,
                    args.parity_mask,
                    pat,
                    tail,
                )
                if cycles < best[0]:
                    best = (cycles, pat, tail)
                    print(f"new best cycles={cycles} pattern={pat} tail={tail}")
    else:
        tail_len = args.pattern_len_tail or args.pattern_len
        for trial in range(args.trials):
            pat = [random.randint(0, args.jitter_max) for _ in range(args.pattern_len)]
            tail = None
            if tail_len != args.pattern_len:
                tail = [random.randint(0, args.jitter_max) for _ in range(tail_len)]
            cycles = build_cycles(
                args.forest_height,
                args.rounds,
                args.batch_size,
                args.groups,
                args.spacing,
                args.tail_spacing,
                args.parity_mask,
                pat,
                tail,
            )
            if cycles < best[0]:
                best = (cycles, pat, tail)
                print(f"new best cycles={cycles} pattern={pat} tail={tail}")
            if (trial + 1) % 100 == 0:
                print(f"trial {trial + 1} best={best[0]}")

    print(f"best cycles={best[0]} pattern={best[1]} tail={best[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
