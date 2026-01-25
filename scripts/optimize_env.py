#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ENV_KEYS = [
    "GROUPS_PER_BATCH",
    "START_SPACING",
    "START_SPACING_TAIL",
    "START_PERIOD",
    "PARITY_MASK",
    "START_JITTER_PATTERN",
    "START_JITTER_PATTERN_TAIL",
    "START_OFFSETS",
    "SCHED_POLICY",
    "SCHED_PRIORITY",
    "PHASE_SPACING",
    "PHASE_SPACING_TAIL",
    "PHASE_JITTER_PATTERN",
    "PHASE_JITTER_PATTERN_TAIL",
]


def parse_range_list(text: str) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            range_part, step_part = (part.split(":", 1) + ["1"])[:2]
            start_str, end_str = range_part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = int(step_part)
            if step <= 0:
                raise ValueError(f"Invalid step in '{part}'")
            if start <= end:
                values.extend(list(range(start, end + 1, step)))
            else:
                values.extend(list(range(start, end - 1, -step)))
        else:
            values.append(int(part))
    return values


def parse_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    return [int(part) for part in parts]


def parse_kv_list(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def import_kernel_builder():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from perf_takehome import KernelBuilder, VLEN

    return KernelBuilder, VLEN


def set_env(env: dict[str, str | None], keys: Iterable[str]) -> None:
    for key in keys:
        value = env.get(key)
        if value is None or value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


def compute_base_offsets(
    total_groups: int,
    groups_per_batch: int,
    spacing: int,
    tail_spacing: int,
    start_period: int,
    jitter_pattern: list[int] | None,
    jitter_pattern_tail: list[int] | None,
) -> list[int]:
    offsets: list[int] = []
    batches = (total_groups + groups_per_batch - 1) // groups_per_batch
    for batch in range(batches):
        batch_groups = min(groups_per_batch, total_groups - batch * groups_per_batch)
        batch_spacing = spacing
        if batch_groups < groups_per_batch:
            batch_spacing = tail_spacing
        pattern = jitter_pattern
        if batch_groups < groups_per_batch and jitter_pattern_tail is not None:
            pattern = jitter_pattern_tail
        for g in range(batch_groups):
            if start_period > 0:
                base = (g % start_period) * batch_spacing
            else:
                base = g * batch_spacing
            if pattern:
                base += pattern[g % len(pattern)]
            offsets.append(base)
    return offsets


@dataclass
class Result:
    cycles: int
    env: dict[str, str | None]


def format_env(env: dict[str, str | None]) -> str:
    ordered_keys = [
        "GROUPS_PER_BATCH",
        "START_SPACING",
        "START_SPACING_TAIL",
        "START_PERIOD",
        "PARITY_MASK",
        "START_JITTER_PATTERN",
        "START_JITTER_PATTERN_TAIL",
        "START_OFFSETS",
        "PHASE_SPACING",
        "PHASE_SPACING_TAIL",
        "PHASE_JITTER_PATTERN",
        "PHASE_JITTER_PATTERN_TAIL",
        "SCHED_POLICY",
        "SCHED_PRIORITY",
    ]
    parts = []
    for key in ordered_keys:
        value = env.get(key)
        if value:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def build_cycles(
    kb_cls,
    forest_height: int,
    rounds: int,
    batch_size: int,
    env: dict[str, str | None],
    env_keys: Iterable[str],
    cache: dict[tuple[tuple[str, str | None], ...], int],
) -> int:
    key = tuple((k, env.get(k)) for k in env_keys)
    if key in cache:
        return cache[key]

    set_env(env, env_keys)
    kb = kb_cls()
    n_nodes = 2 ** (forest_height + 1) - 1
    try:
        kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
        cycles = len(kb.instrs)
    except Exception:
        cycles = 1 << 30

    cache[key] = cycles
    return cycles


def _cache_key(env: dict[str, str | None], env_keys: Iterable[str]) -> tuple[tuple[str, str | None], ...]:
    return tuple((k, env.get(k)) for k in env_keys)


def _run_combo(
    combo: tuple[int, int, int, int, int, int, int, str, str],
    kb_cls,
    forest_height: int,
    rounds: int,
    batch_size: int,
    env_keys: Iterable[str],
    base_env: dict[str, str | None],
    cache: dict[tuple[tuple[str, str | None], ...], int],
) -> tuple[int, dict[str, str | None]]:
    (
        groups,
        spacing,
        tail_spacing,
        period,
        parity,
        phase_spacing,
        phase_spacing_tail,
        sched_policy,
        sched_priority,
    ) = combo
    env = dict(base_env)
    env["GROUPS_PER_BATCH"] = str(groups)
    env["START_SPACING"] = str(spacing)
    env["START_SPACING_TAIL"] = str(tail_spacing)
    env["START_PERIOD"] = str(period)
    env["PARITY_MASK"] = str(parity)
    env["PHASE_SPACING"] = str(phase_spacing)
    env["PHASE_SPACING_TAIL"] = str(phase_spacing_tail)
    env["SCHED_POLICY"] = sched_policy
    env["SCHED_PRIORITY"] = sched_priority
    cycles = build_cycles(
        kb_cls,
        forest_height,
        rounds,
        batch_size,
        env,
        env_keys,
        cache,
    )
    return cycles, env


_WORKER_STATE: dict[str, object] = {}


def _init_worker(
    kb_cls,
    forest_height: int,
    rounds: int,
    batch_size: int,
    env_keys: list[str],
    base_env: dict[str, str | None],
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "kb_cls": kb_cls,
        "forest_height": forest_height,
        "rounds": rounds,
        "batch_size": batch_size,
        "env_keys": list(env_keys),
        "base_env": dict(base_env),
        "cache": {},
    }


def _run_combo_worker(combo: tuple[int, int, int, int, int, int, int, str, str]) -> tuple[int, dict[str, str | None]]:
    state = _WORKER_STATE
    return _run_combo(
        combo,
        state["kb_cls"],
        state["forest_height"],
        state["rounds"],
        state["batch_size"],
        state["env_keys"],
        state["base_env"],
        state["cache"],
    )


def grid_search(
    kb_cls,
    forest_height: int,
    rounds: int,
    batch_size: int,
    env_keys: list[str],
    base_env: dict[str, str | None],
    groups_list: list[int],
    spacing_list: list[int],
    tail_spacing_list: list[int],
    period_list: list[int],
    parity_masks: list[int],
    phase_spacing_list: list[int],
    phase_spacing_tail_list: list[int],
    sched_policies: list[str],
    sched_priorities: list[str],
    log_every: int,
    jobs: int,
    cache: dict[tuple[tuple[str, str | None], ...], int],
) -> Result:
    best = None
    total = (
        len(groups_list)
        * len(spacing_list)
        * len(tail_spacing_list)
        * len(period_list)
        * len(parity_masks)
        * len(phase_spacing_list)
        * len(phase_spacing_tail_list)
        * len(sched_policies)
        * len(sched_priorities)
    )
    seen = 0

    combos = itertools.product(
        groups_list,
        spacing_list,
        tail_spacing_list,
        period_list,
        parity_masks,
        phase_spacing_list,
        phase_spacing_tail_list,
        sched_policies,
        sched_priorities,
    )

    if jobs <= 1:
        for combo in combos:
            cycles, env = _run_combo(
                combo,
                kb_cls,
                forest_height,
                rounds,
                batch_size,
                env_keys,
                base_env,
                cache,
            )
            cache[_cache_key(env, env_keys)] = cycles
            seen += 1
            if best is None or cycles < best.cycles:
                best = Result(cycles=cycles, env=env)
                print(f"new best {cycles} ({seen}/{total}) {format_env(env)}", flush=True)
            if log_every > 0 and seen % log_every == 0:
                best_line = format_env(best.env) if best is not None else ""
                best_cycles = best.cycles if best is not None else "n/a"
                print(
                    f"progress {seen}/{total} best={best_cycles} {best_line}".rstrip(),
                    flush=True,
                )
    else:
        ctx = mp.get_context("spawn")

        # Use a conservative chunksize to balance overhead and progress updates.
        chunksize = max(1, total // (jobs * 50))
        if chunksize > 100:
            chunksize = 100
        with ctx.Pool(
            processes=jobs,
            initializer=_init_worker,
            initargs=(kb_cls, forest_height, rounds, batch_size, env_keys, base_env),
        ) as pool:
            for cycles, env in pool.imap_unordered(
                _run_combo_worker,
                combos,
                chunksize=chunksize,
            ):
                cache[_cache_key(env, env_keys)] = cycles
                seen += 1
                if best is None or cycles < best.cycles:
                    best = Result(cycles=cycles, env=env)
                    print(
                        f"new best {cycles} ({seen}/{total}) {format_env(env)}",
                        flush=True,
                    )
                if log_every > 0 and seen % log_every == 0:
                    best_line = format_env(best.env) if best is not None else ""
                    best_cycles = best.cycles if best is not None else "n/a"
                    print(
                        f"progress {seen}/{total} best={best_cycles} {best_line}".rstrip(),
                        flush=True,
                    )
    if best is None:
        raise RuntimeError("No grid search results")
    return best


def jitter_search(
    kb_cls,
    forest_height: int,
    rounds: int,
    batch_size: int,
    env_keys: list[str],
    base_result: Result,
    jitter_len: int,
    jitter_tail_len: int | None,
    jitter_max: int,
    jitter_trials: int,
    jitter_mode: str,
    seed: int | None,
    cache: dict[tuple[tuple[str, str | None], ...], int],
) -> Result:
    if jitter_mode not in ("random", "exhaustive"):
        raise ValueError(f"Unknown jitter mode '{jitter_mode}'")

    if seed is not None:
        random.seed(seed)

    best = Result(cycles=base_result.cycles, env=dict(base_result.env))
    tail_len = jitter_tail_len or jitter_len
    if jitter_mode == "exhaustive":
        values = list(range(jitter_max + 1))
        for pat in itertools.product(values, repeat=jitter_len):
            if tail_len == jitter_len:
                patterns = [(list(pat), None)]
            else:
                patterns = [
                    (list(pat), list(tail))
                    for tail in itertools.product(values, repeat=tail_len)
                ]
            for pattern, pattern_tail in patterns:
                env = dict(base_result.env)
                env["START_JITTER_PATTERN"] = ",".join(str(v) for v in pattern)
                env["START_JITTER_PATTERN_TAIL"] = (
                    ",".join(str(v) for v in pattern_tail) if pattern_tail else None
                )
                cycles = build_cycles(
                    kb_cls,
                    forest_height,
                    rounds,
                    batch_size,
                    env,
                    env_keys,
                    cache,
                )
                if cycles < best.cycles:
                    best = Result(cycles=cycles, env=env)
                    print(f"new best jitter {cycles} {format_env(env)}")
    else:
        for trial in range(jitter_trials):
            pattern = [random.randint(0, jitter_max) for _ in range(jitter_len)]
            pattern_tail = None
            if tail_len != jitter_len:
                pattern_tail = [random.randint(0, jitter_max) for _ in range(tail_len)]
            env = dict(base_result.env)
            env["START_JITTER_PATTERN"] = ",".join(str(v) for v in pattern)
            env["START_JITTER_PATTERN_TAIL"] = (
                ",".join(str(v) for v in pattern_tail) if pattern_tail else None
            )
            cycles = build_cycles(
                kb_cls,
                forest_height,
                rounds,
                batch_size,
                env,
                env_keys,
                cache,
            )
            if cycles < best.cycles:
                best = Result(cycles=cycles, env=env)
                print(f"new best jitter {cycles} {format_env(env)}")
            if (trial + 1) % 50 == 0:
                print(f"jitter trial {trial + 1}/{jitter_trials} best={best.cycles}")
    return best


def offset_search(
    kb_cls,
    forest_height: int,
    rounds: int,
    batch_size: int,
    env_keys: list[str],
    base_result: Result,
    offset_trials: int,
    offset_max_delta: int,
    offset_max: int,
    offset_seed: int | None,
    cache: dict[tuple[tuple[str, str | None], ...], int],
) -> Result:
    if offset_trials <= 0:
        return base_result

    if offset_seed is not None:
        random.seed(offset_seed)

    KernelBuilder, VLEN = import_kernel_builder()
    total_groups = batch_size // VLEN

    groups_per_batch = int(base_result.env.get("GROUPS_PER_BATCH", "1"))
    spacing = int(base_result.env.get("START_SPACING", "0"))
    tail_spacing = int(base_result.env.get("START_SPACING_TAIL", str(spacing)))
    start_period = int(base_result.env.get("START_PERIOD", "0"))
    jitter_pattern = parse_list(base_result.env.get("START_JITTER_PATTERN"))
    jitter_pattern_tail = parse_list(base_result.env.get("START_JITTER_PATTERN_TAIL"))
    base_offsets = compute_base_offsets(
        total_groups,
        groups_per_batch,
        spacing,
        tail_spacing,
        start_period,
        jitter_pattern,
        jitter_pattern_tail,
    )

    best_offsets = list(base_offsets)
    best = Result(cycles=base_result.cycles, env=dict(base_result.env))
    best.env["START_OFFSETS"] = ",".join(str(v) for v in best_offsets)
    best.cycles = build_cycles(
        kb_cls,
        forest_height,
        rounds,
        batch_size,
        best.env,
        env_keys,
        cache,
    )

    for trial in range(offset_trials):
        idx = random.randrange(len(best_offsets))
        delta = random.randint(-offset_max_delta, offset_max_delta)
        if delta == 0:
            continue
        candidate = list(best_offsets)
        candidate[idx] = max(0, min(offset_max, candidate[idx] + delta))
        env = dict(base_result.env)
        env["START_OFFSETS"] = ",".join(str(v) for v in candidate)
        cycles = build_cycles(
            kb_cls,
            forest_height,
            rounds,
            batch_size,
            env,
            env_keys,
            cache,
        )
        if cycles < best.cycles:
            best_offsets = candidate
            best = Result(cycles=cycles, env=env)
            print(f"new best offsets {cycles} {format_env(env)}")
        if (trial + 1) % 50 == 0:
            print(f"offset trial {trial + 1}/{offset_trials} best={best.cycles}")

    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Search env settings for best kernel cycles.")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--groups", help="Groups per batch range (e.g. 16-21)")
    parser.add_argument("--spacing", help="Start spacing range (e.g. 5-8)")
    parser.add_argument("--tail-spacing", help="Tail spacing range (e.g. 9-12)")
    parser.add_argument("--period", help="Start period range (e.g. 0,4)")
    parser.add_argument("--parity-masks", help="Parity mask range (e.g. 0-255:4)")
    parser.add_argument("--phase-spacing", help="Phase spacing range (e.g. 0,4)")
    parser.add_argument("--phase-spacing-tail", help="Phase spacing tail range")
    parser.add_argument("--sched-policy", help="Scheduler policies (e.g. greedy,balanced)")
    parser.add_argument("--sched-priority", help="Scheduler priorities list")
    parser.add_argument("--fixed-env", action="append", default=[], help="Fixed env vars KEY=VALUE")
    parser.add_argument("--seed", type=int, help="Random seed for jitter/offset search")
    parser.add_argument("--jitter-mode", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--jitter-len", type=int, default=4)
    parser.add_argument("--jitter-tail-len", type=int)
    parser.add_argument("--jitter-max", type=int, default=6)
    parser.add_argument("--jitter-trials", type=int, default=100)
    parser.add_argument("--no-jitter", action="store_true", help="Skip jitter search")
    parser.add_argument("--offset-trials", type=int, default=100)
    parser.add_argument("--offset-max-delta", type=int, default=15)
    parser.add_argument("--offset-max", type=int, default=150)
    parser.add_argument("--no-offset-search", action="store_true", help="Skip offset search")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=1)

    args = parser.parse_args()

    KernelBuilder, VLEN = import_kernel_builder()

    if args.groups:
        groups_list = parse_range_list(args.groups)
    else:
        groups_list = (
            parse_range_list("16-21")
            if args.mode == "full"
            else parse_range_list("19-21")
        )

    if args.spacing:
        spacing_list = parse_range_list(args.spacing)
    else:
        spacing_list = (
            parse_range_list("4-10")
            if args.mode == "full"
            else parse_range_list("6-8")
        )

    if args.tail_spacing:
        tail_spacing_list = parse_range_list(args.tail_spacing)
    else:
        tail_spacing_list = (
            parse_range_list("6-14")
            if args.mode == "full"
            else parse_range_list("10-12")
        )

    if args.period:
        period_list = parse_range_list(args.period)
    else:
        period_list = [0]

    if args.parity_masks:
        parity_masks = parse_range_list(args.parity_masks)
    else:
        parity_masks = (
            parse_range_list("0-255") if args.mode == "full" else parse_range_list("0-255:16")
        )

    if args.phase_spacing:
        phase_spacing_list = parse_range_list(args.phase_spacing)
    else:
        phase_spacing_list = [0]

    if args.phase_spacing_tail:
        phase_spacing_tail_list = parse_range_list(args.phase_spacing_tail)
    else:
        phase_spacing_tail_list = [0]

    if args.sched_policy:
        sched_policies = [part.strip() for part in args.sched_policy.split(",") if part.strip()]
    else:
        sched_policies = ["greedy"]

    if args.sched_priority:
        sched_priorities = [part.strip() for part in args.sched_priority.split(",") if part.strip()]
    else:
        sched_priorities = [""]

    fixed_env = parse_kv_list(args.fixed_env)
    for key in fixed_env:
        if key not in ENV_KEYS:
            ENV_KEYS.append(key)

    base_env: dict[str, str | None] = dict(fixed_env)

    cache: dict[tuple[tuple[str, str | None], ...], int] = {}

    best = grid_search(
        KernelBuilder,
        args.forest_height,
        args.rounds,
        args.batch_size,
        ENV_KEYS,
        base_env,
        groups_list,
        spacing_list,
        tail_spacing_list,
        period_list,
        parity_masks,
        phase_spacing_list,
        phase_spacing_tail_list,
        sched_policies,
        sched_priorities,
        args.log_every,
        args.jobs,
        cache,
    )

    if not args.no_jitter:
        best = jitter_search(
            KernelBuilder,
            args.forest_height,
            args.rounds,
            args.batch_size,
            ENV_KEYS,
            best,
            args.jitter_len,
            args.jitter_tail_len,
            args.jitter_max,
            args.jitter_trials,
            args.jitter_mode,
            args.seed,
            cache,
        )

    if not args.no_offset_search:
        best = offset_search(
            KernelBuilder,
            args.forest_height,
            args.rounds,
            args.batch_size,
            ENV_KEYS,
            best,
            args.offset_trials,
            args.offset_max_delta,
            args.offset_max,
            args.seed,
            cache,
        )

    print(f"\nBest cycles: {best.cycles}")
    print(format_env(best.env))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
