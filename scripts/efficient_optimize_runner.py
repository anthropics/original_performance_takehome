#!/usr/bin/env python3
"""
Purpose:
  Staged optimizer runner for kernel env tuning. Runs a coarse grid search,
  optionally refines around the best point, then optionally runs jitter and
  offset searches to squeeze out more cycles. Uses optimize_env helpers.

Args:
  --forest-height: Forest height passed to KernelBuilder.
  --rounds: Number of rounds per input.
  --batch-size: Batch size for the kernel build.
  --jobs: Number of worker processes for the grid search.
  --log-every: Print progress every N grid evaluations.
  --coarse-groups: GROUPS_PER_BATCH range for coarse grid (e.g. 18-21).
  --coarse-spacing: START_SPACING range for coarse grid.
  --coarse-tail-spacing: START_SPACING_TAIL range for coarse grid.
  --coarse-period: START_PERIOD range for coarse grid.
  --coarse-parity: PARITY_MASK range for coarse grid (e.g. 0-255:32).
  --coarse-phase-spacing: PHASE_SPACING range for coarse grid.
  --coarse-phase-spacing-tail: PHASE_SPACING_TAIL range for coarse grid.
  --sched-policy: Comma list of scheduler policies to test.
  --sched-priority: Comma list of scheduler priorities to test.
  --fixed-env: Fixed env vars KEY=VALUE (overrides any grid/refine list).
  --skip-refine: Skip the refine grid stage after coarse search.
  --refine-groups-delta: +/- delta around best GROUPS_PER_BATCH.
  --refine-spacing-delta: +/- delta around best START_SPACING.
  --refine-tail-spacing-delta: +/- delta around best START_SPACING_TAIL.
  --refine-period-delta: +/- delta around best START_PERIOD.
  --parity-refine-span: Parity mask +/- span around best (0..255).
  --parity-refine-step: Step for parity refinement around best.
  --refine-phase-spacing-delta: +/- delta around best PHASE_SPACING.
  --refine-phase-spacing-tail-delta: +/- delta around best PHASE_SPACING_TAIL.
  --seed: Random seed for jitter and offset search.
  --jitter-mode: "random" or "exhaustive" jitter search mode.
  --jitter-len: Length of START_JITTER_PATTERN.
  --jitter-tail-len: Length of START_JITTER_PATTERN_TAIL (defaults to jitter-len).
  --jitter-max: Max jitter element value.
  --jitter-trials: Number of random jitter trials.
  --no-jitter: Skip jitter search.
  --offset-trials: Number of random offset trials.
  --offset-max-delta: Max delta applied per offset trial.
  --offset-max: Clamp offset values to [0, offset-max].
  --no-offset-search: Skip offset search.

Examples:
  python3 scripts/efficient_optimize_runner.py --jobs 8 --no-jitter --no-offset-search
  python3 scripts/efficient_optimize_runner.py --jobs 8 --fixed-env SCHED_POLICY=beam
  python3 scripts/efficient_optimize_runner.py --jobs 8 --coarse-parity 0-255:16 --jitter-trials 200
  nohup python3 scripts/efficient_optimize_runner.py --jobs 4 \
    --coarse-groups 19-21 \
    --coarse-spacing 6-8 \
    --coarse-tail-spacing 10-12 \
    --coarse-period 0 \
    --coarse-parity 64-76:2 \
    --coarse-phase-spacing 0 \
    --coarse-phase-spacing-tail 0 \
    --no-jitter \
    --no-offset-search \
    --fixed-env START_OFFSETS=2,8,14,23,33,36,42,51,58,64,70,79,100,92,98,107,106,120,126,135,1,12,22,35,52,56,66,88,90,100,110,123 \
    --fixed-env START_JITTER_PATTERN=2,1,0,2 > logs/opt_coarse_jobs4.log 2>&1 &
"""
from __future__ import annotations

import argparse
from typing import Iterable

import optimize_env


def _parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _env_int(env: dict[str, str | None], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _override_list(
    env: dict[str, str | None],
    key: str,
    values: list[int],
) -> list[int]:
    if key in env and env[key] not in (None, ""):
        return [int(env[key])]
    return values


def _override_str_list(
    env: dict[str, str | None],
    key: str,
    values: list[str],
) -> list[str]:
    if key in env and env[key] not in (None, ""):
        return [str(env[key])]
    return values


def _refine_range(value: int, delta: int, min_value: int = 0, max_value: int | None = None) -> list[int]:
    if delta <= 0:
        return [value]
    start = max(min_value, value - delta)
    end = value + delta if max_value is None else min(max_value, value + delta)
    return list(range(start, end + 1))


def _refine_parity(value: int, span: int, step: int) -> list[int]:
    if span <= 0 or step <= 0:
        return [value]
    values = []
    for delta in range(-span, span + 1, step):
        candidate = value + delta
        if 0 <= candidate <= 255:
            values.append(candidate)
    return sorted(set(values))


def _print_stage(label: str, result: optimize_env.Result) -> None:
    print(f"\n{label} cycles: {result.cycles}")
    print(optimize_env.format_env(result.env))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a staged optimization search (coarse -> refine -> jitter/offset)."
    )
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=500)

    parser.add_argument("--coarse-groups", default="18-21")
    parser.add_argument("--coarse-spacing", default="6-8")
    parser.add_argument("--coarse-tail-spacing", default="10-12")
    parser.add_argument("--coarse-period", default="0")
    parser.add_argument("--coarse-parity", default="0-255:32")
    parser.add_argument("--coarse-phase-spacing", default="0")
    parser.add_argument("--coarse-phase-spacing-tail", default="0")

    parser.add_argument("--sched-policy", default="greedy")
    parser.add_argument("--sched-priority", default="")
    parser.add_argument("--fixed-env", action="append", default=[], help="Fixed env vars KEY=VALUE")

    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--refine-groups-delta", type=int, default=1)
    parser.add_argument("--refine-spacing-delta", type=int, default=1)
    parser.add_argument("--refine-tail-spacing-delta", type=int, default=1)
    parser.add_argument("--refine-period-delta", type=int, default=0)
    parser.add_argument("--parity-refine-span", type=int, default=6)
    parser.add_argument("--parity-refine-step", type=int, default=2)
    parser.add_argument("--refine-phase-spacing-delta", type=int, default=1)
    parser.add_argument("--refine-phase-spacing-tail-delta", type=int, default=1)

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

    args = parser.parse_args()

    KernelBuilder, _ = optimize_env.import_kernel_builder()

    env_keys: list[str] = list(optimize_env.ENV_KEYS)
    fixed_env = optimize_env.parse_kv_list(args.fixed_env)
    for key in fixed_env:
        if key not in env_keys:
            env_keys.append(key)

    base_env: dict[str, str | None] = dict(fixed_env)

    groups_list = optimize_env.parse_range_list(args.coarse_groups)
    spacing_list = optimize_env.parse_range_list(args.coarse_spacing)
    tail_spacing_list = optimize_env.parse_range_list(args.coarse_tail_spacing)
    period_list = optimize_env.parse_range_list(args.coarse_period)
    parity_masks = optimize_env.parse_range_list(args.coarse_parity)
    phase_spacing_list = optimize_env.parse_range_list(args.coarse_phase_spacing)
    phase_spacing_tail_list = optimize_env.parse_range_list(args.coarse_phase_spacing_tail)
    sched_policies = _parse_csv(args.sched_policy) or ["greedy"]
    sched_priorities = _parse_csv(args.sched_priority) or [""]

    groups_list = _override_list(base_env, "GROUPS_PER_BATCH", groups_list)
    spacing_list = _override_list(base_env, "START_SPACING", spacing_list)
    tail_spacing_list = _override_list(base_env, "START_SPACING_TAIL", tail_spacing_list)
    period_list = _override_list(base_env, "START_PERIOD", period_list)
    parity_masks = _override_list(base_env, "PARITY_MASK", parity_masks)
    phase_spacing_list = _override_list(base_env, "PHASE_SPACING", phase_spacing_list)
    phase_spacing_tail_list = _override_list(
        base_env, "PHASE_SPACING_TAIL", phase_spacing_tail_list
    )
    sched_policies = _override_str_list(base_env, "SCHED_POLICY", sched_policies)
    sched_priorities = _override_str_list(base_env, "SCHED_PRIORITY", sched_priorities)

    cache: dict[tuple[tuple[str, str | None], ...], int] = {}

    coarse = optimize_env.grid_search(
        KernelBuilder,
        args.forest_height,
        args.rounds,
        args.batch_size,
        env_keys,
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
    _print_stage("Coarse best", coarse)

    best = coarse
    if not args.skip_refine:
        best_env = best.env
        refine_groups = _refine_range(
            _env_int(best_env, "GROUPS_PER_BATCH", groups_list[0]),
            args.refine_groups_delta,
            min_value=1,
        )
        refine_spacing = _refine_range(
            _env_int(best_env, "START_SPACING", spacing_list[0]),
            args.refine_spacing_delta,
            min_value=0,
        )
        refine_tail_spacing = _refine_range(
            _env_int(best_env, "START_SPACING_TAIL", tail_spacing_list[0]),
            args.refine_tail_spacing_delta,
            min_value=0,
        )
        refine_period = _refine_range(
            _env_int(best_env, "START_PERIOD", period_list[0]),
            args.refine_period_delta,
            min_value=0,
        )
        refine_parity = _refine_parity(
            _env_int(best_env, "PARITY_MASK", parity_masks[0]),
            args.parity_refine_span,
            args.parity_refine_step,
        )
        refine_phase_spacing = _refine_range(
            _env_int(best_env, "PHASE_SPACING", phase_spacing_list[0]),
            args.refine_phase_spacing_delta,
            min_value=0,
        )
        refine_phase_spacing_tail = _refine_range(
            _env_int(best_env, "PHASE_SPACING_TAIL", phase_spacing_tail_list[0]),
            args.refine_phase_spacing_tail_delta,
            min_value=0,
        )

        refine_groups = _override_list(base_env, "GROUPS_PER_BATCH", refine_groups)
        refine_spacing = _override_list(base_env, "START_SPACING", refine_spacing)
        refine_tail_spacing = _override_list(base_env, "START_SPACING_TAIL", refine_tail_spacing)
        refine_period = _override_list(base_env, "START_PERIOD", refine_period)
        refine_parity = _override_list(base_env, "PARITY_MASK", refine_parity)
        refine_phase_spacing = _override_list(base_env, "PHASE_SPACING", refine_phase_spacing)
        refine_phase_spacing_tail = _override_list(
            base_env, "PHASE_SPACING_TAIL", refine_phase_spacing_tail
        )

        best = optimize_env.grid_search(
            KernelBuilder,
            args.forest_height,
            args.rounds,
            args.batch_size,
            env_keys,
            base_env,
            refine_groups,
            refine_spacing,
            refine_tail_spacing,
            refine_period,
            refine_parity,
            refine_phase_spacing,
            refine_phase_spacing_tail,
            sched_policies,
            sched_priorities,
            args.log_every,
            args.jobs,
            cache,
        )
        _print_stage("Refine best", best)

    if not args.no_jitter:
        best = optimize_env.jitter_search(
            KernelBuilder,
            args.forest_height,
            args.rounds,
            args.batch_size,
            env_keys,
            best,
            args.jitter_len,
            args.jitter_tail_len,
            args.jitter_max,
            args.jitter_trials,
            args.jitter_mode,
            args.seed,
            cache,
        )
        _print_stage("Jitter best", best)

    if not args.no_offset_search:
        best = optimize_env.offset_search(
            KernelBuilder,
            args.forest_height,
            args.rounds,
            args.batch_size,
            env_keys,
            best,
            args.offset_trials,
            args.offset_max_delta,
            args.offset_max,
            args.seed,
            cache,
        )
        _print_stage("Offset best", best)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
