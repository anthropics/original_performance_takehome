#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
TESTS_DIR = os.path.join(ROOT, "tests")
sys.path.insert(0, ROOT)
sys.path.insert(0, TESTS_DIR)

from frozen_problem import Tree, Input, build_mem_image, Machine, N_CORES
from perf_takehome import KernelBuilder


def run_cycles(mem_base, forest_height, n_nodes, batch_size, rounds, mask):
    os.environ["PARITY_MASK"] = str(mask)
    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    machine = Machine(mem_base[:], kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine.cycle


def main():
    start = int(sys.argv[1], 0) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2], 0) if len(sys.argv) > 2 else 256

    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem_base = build_mem_image(forest, inp)

    best_cycles = None
    best_mask = None
    for mask in range(start, end):
        cycles = run_cycles(mem_base, forest.height, len(forest.values), len(inp.indices), 16, mask)
        if best_cycles is None or cycles < best_cycles:
            best_cycles = cycles
            best_mask = mask
        print(f"{mask:3d} cycles={cycles} best={best_cycles} mask={best_mask}")

    print(f"BEST mask={best_mask} cycles={best_cycles}")


if __name__ == "__main__":
    main()
