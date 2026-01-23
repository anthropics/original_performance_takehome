import itertools
import os
import random
import sys

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
    seed: int = 123,
    verify: bool = True,
) -> int:
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder(
        unroll=unroll,
        load_interleave=load_interleave,
        compute_schedule=compute_schedule,
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


def main():
    forest_height = 10
    rounds = 16
    batch_size = 256

    unroll_options = [4, 6, 8]
    load_interleave_options = [1, 2, 3]
    schedule_options = ["stagewise", "blockwise"]

    results = []
    for unroll, load_interleave, schedule in itertools.product(
        unroll_options, load_interleave_options, schedule_options
    ):
        cycle = run_cycles(
            forest_height,
            rounds,
            batch_size,
            unroll=unroll,
            load_interleave=load_interleave,
            compute_schedule=schedule,
        )
        results.append((cycle, unroll, load_interleave, schedule))
        speedup = BASELINE / cycle
        print(
            f"unroll={unroll} load_interleave={load_interleave} schedule={schedule} "
            f"cycles={cycle} speedup={speedup:.2f}x"
        )

    results.sort()
    best_cycle, best_unroll, best_interleave, best_schedule = results[0]
    print("\nBest:")
    print(
        f"unroll={best_unroll} load_interleave={best_interleave} "
        f"schedule={best_schedule} cycles={best_cycle}"
    )


if __name__ == "__main__":
    main()
