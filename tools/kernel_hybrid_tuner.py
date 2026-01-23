import argparse
import csv
import itertools
import os
import random
import sys
from datetime import datetime

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from problem import (
    Machine,
    Tree,
    Input,
    N_CORES,
    reference_kernel2,
    build_mem_image,
)
from perf_takehome import KernelBuilder, BASELINE


def run_cycles(
    forest_height: int,
    rounds: int,
    batch_size: int,
    unroll: int,
    load_interleave: int,
    compute_schedule: str,
    hash_interleave: tuple[int, int, int],
    block_group: int,
    gather_strategy: str,
    seed: int,
    verify: bool,
) -> int:
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder(
        unroll=unroll,
        load_interleave=load_interleave,
        compute_schedule=compute_schedule,
        hash_interleave=hash_interleave,
        block_group=block_group,
        gather_strategy=gather_strategy,
    )
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    if verify:
        for ref_mem in reference_kernel2(mem):
            pass
        inp_values_p = ref_mem[6]
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), "Incorrect output values"

    return machine.cycle


def iter_grid(
    unroll_options: list[int],
    load_interleave_options: list[int],
    hash_interleave_values: list[int],
    schedule_options: list[str],
    block_group_specs: list[str],
    gather_options: list[str],
):
    for unroll, load_interleave, schedule in itertools.product(
        unroll_options, load_interleave_options, schedule_options
    ):
        block_groups: list[int] = []
        for spec in block_group_specs:
            if spec == "unroll":
                block_groups.append(unroll)
            else:
                block_groups.append(int(spec))
        block_groups = sorted({bg for bg in block_groups if bg > 0 and bg <= unroll})
        for block_group in block_groups:
            for gather_strategy in gather_options:
                for h1 in hash_interleave_values:
                    for h2 in hash_interleave_values:
                        for h3 in hash_interleave_values:
                            yield (
                                unroll,
                                load_interleave,
                                schedule,
                                (h1, h2, h3),
                                block_group,
                                gather_strategy,
                            )


def iter_random(
    unroll_options: list[int],
    load_interleave_options: list[int],
    hash_interleave_values: list[int],
    schedule_options: list[str],
    block_group_specs: list[str],
    gather_options: list[str],
    trials: int,
    seed: int,
):
    rng = random.Random(seed)
    for _ in range(trials):
        unroll = rng.choice(unroll_options)
        load_interleave = rng.choice(load_interleave_options)
        schedule = rng.choice(schedule_options)
        block_groups: list[int] = []
        for spec in block_group_specs:
            if spec == "unroll":
                block_groups.append(unroll)
            else:
                block_groups.append(int(spec))
        block_groups = [bg for bg in block_groups if bg > 0 and bg <= unroll]
        block_group = rng.choice(block_groups)
        gather_strategy = rng.choice(gather_options)
        hash_interleave = (
            rng.choice(hash_interleave_values),
            rng.choice(hash_interleave_values),
            rng.choice(hash_interleave_values),
        )
        yield unroll, load_interleave, schedule, hash_interleave, block_group, gather_strategy


def main():
    parser = argparse.ArgumentParser(description="Hybrid brute-force tuner for kernel scheduling.")
    parser.add_argument("--mode", choices=["grid", "random"], default="grid")
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--verify", action="store_true", default=False)
    parser.add_argument("--output", default="hybrid_tuning_results.csv")
    parser.add_argument("--unroll", default="4,5,6,7,8")
    parser.add_argument("--load-interleave", default="1,2,3")
    parser.add_argument("--hash-values", default="0,1,2,3")
    parser.add_argument("--block-groups", default="1,2,3,4,unroll")
    parser.add_argument("--schedule", default="stagewise")
    parser.add_argument("--gather-strategy", default="by_buffer")
    args = parser.parse_args()

    forest_height = 10
    rounds = 16
    batch_size = 256

    def parse_int_list(value: str) -> list[int]:
        if not value:
            return []
        return [int(chunk) for chunk in value.split(",") if chunk.strip()]

    def parse_schedule_list(value: str) -> list[str]:
        if not value:
            return []
        return [chunk.strip().lower() for chunk in value.split(",") if chunk.strip()]

    def parse_str_list(value: str) -> list[str]:
        if not value:
            return []
        return [chunk.strip().lower() for chunk in value.split(",") if chunk.strip()]

    unroll_options = parse_int_list(args.unroll)
    load_interleave_options = parse_int_list(args.load_interleave)
    hash_interleave_values = parse_int_list(args.hash_values)
    schedule_options = parse_schedule_list(args.schedule)
    gather_options = parse_str_list(args.gather_strategy)

    if not unroll_options or not load_interleave_options or not hash_interleave_values:
        raise ValueError("unroll, load-interleave, and hash-values must be non-empty")
    if not schedule_options:
        schedule_options = ["stagewise"]
    if not gather_options:
        gather_options = ["by_buffer"]

    raw_block_groups = [chunk.strip().lower() for chunk in args.block_groups.split(",") if chunk.strip()]
    block_group_specs = []
    for chunk in raw_block_groups:
        if chunk == "unroll":
            block_group_specs.append(chunk)
        else:
            int(chunk)
            block_group_specs.append(chunk)
    if not block_group_specs:
        block_group_specs = ["unroll"]

    if args.mode == "grid":
        iterator = iter_grid(
            unroll_options,
            load_interleave_options,
            hash_interleave_values,
            schedule_options,
            block_group_specs,
            gather_options,
        )
    else:
        iterator = iter_random(
            unroll_options,
            load_interleave_options,
            hash_interleave_values,
            schedule_options,
            block_group_specs,
            gather_options,
            trials=args.trials,
            seed=args.seed,
        )

    top_results: list[tuple[int, int, int, str, tuple[int, int, int], int, str]] = []

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "timestamp",
                "cycles",
                "unroll",
                "load_interleave",
                "schedule",
                "hash_interleave",
                "block_group",
                "gather_strategy",
                "speedup",
            ]
        )
        for (
            unroll,
            load_interleave,
            schedule,
            hash_interleave,
            block_group,
            gather_strategy,
        ) in iterator:
            cycle = run_cycles(
                forest_height,
                rounds,
                batch_size,
                unroll=unroll,
                load_interleave=load_interleave,
                compute_schedule=schedule,
                hash_interleave=hash_interleave,
                block_group=block_group,
                gather_strategy=gather_strategy,
                seed=args.seed,
                verify=args.verify,
            )
            speedup = BASELINE / cycle
            stamp = datetime.now().isoformat(timespec="seconds")
            writer.writerow(
                [
                    stamp,
                    cycle,
                    unroll,
                    load_interleave,
                    schedule,
                    f"{hash_interleave}",
                    block_group,
                    gather_strategy,
                    f"{speedup:.2f}",
                ]
            )
            csvfile.flush()

            top_results.append(
                (
                    cycle,
                    unroll,
                    load_interleave,
                    schedule,
                    hash_interleave,
                    block_group,
                    gather_strategy,
                )
            )
            top_results.sort(key=lambda row: row[0])
            top_results = top_results[:10]

            print(
                "unroll={} load_interleave={} schedule={} hash_interleave={} block_group={} "
                "gather_strategy={} cycles={} speedup={:.2f}x".format(
                    unroll,
                    load_interleave,
                    schedule,
                    hash_interleave,
                    block_group,
                    gather_strategy,
                    cycle,
                    speedup,
                )
            )

    print("\nTop 10:")
    for (
        cycle,
        unroll,
        load_interleave,
        schedule,
        hash_interleave,
        block_group,
        gather_strategy,
    ) in top_results:
        print(
            "cycles={} unroll={} load_interleave={} schedule={} hash_interleave={} block_group={} "
            "gather_strategy={}".format(
                cycle,
                unroll,
                load_interleave,
                schedule,
                hash_interleave,
                block_group,
                gather_strategy,
            )
        )


if __name__ == "__main__":
    main()
