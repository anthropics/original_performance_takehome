import argparse
import csv
import importlib
import random
import sys
from collections import defaultdict
from pathlib import Path


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


def import_problem_modules():
    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "tests"
    sys.path.insert(0, str(tests_dir))
    sys.path.insert(0, str(repo_root))
    from frozen_problem import (
        Machine,
        build_mem_image,
        reference_kernel2,
        Tree,
        Input,
        N_CORES,
    )

    return Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES


def run_kernel_test(
    kernel_module: str,
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int | None,
    verify: bool,
):
    Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES = import_problem_modules()

    if seed is not None:
        random.seed(seed)

    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    module = importlib.import_module(kernel_module)
    kb = module.KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    if verify:
        for ref_mem in reference_kernel2(mem):
            pass
        inp_values_p = ref_mem[6]
        if (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            != ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ):
            raise AssertionError("Incorrect output values")

    return machine.cycle


def run_kernel_test_from_tests(forest_height: int, rounds: int, batch_size: int, seed: int | None):
    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "tests"
    sys.path.insert(0, str(tests_dir))
    sys.path.insert(0, str(repo_root))
    import submission_tests

    if seed is not None:
        submission_tests.random.seed(seed)

    return submission_tests.do_kernel_test(forest_height, rounds, batch_size)


def summarize(results: list[dict]):
    by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row in results:
        if row["ok"]:
            by_key[(row["forest_height"], row["rounds"], row["batch_size"])].append(
                row["cycle"]
            )

    summary_rows = []
    for key, cycles in sorted(by_key.items()):
        avg_cycle = sum(cycles) / len(cycles)
        summary_rows.append(
            {
                "forest_height": key[0],
                "rounds": key[1],
                "batch_size": key[2],
                "runs": len(cycles),
                "avg_cycle": avg_cycle,
                "max_cycle": max(cycles),
                "min_cycle": min(cycles),
            }
        )

    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Sweep kernel cycles across a parameter grid.")
    parser.add_argument("--kernel", default="perf_takehome", help="Kernel module to test")
    parser.add_argument("--depths", default="10", help="Forest heights (e.g. 8-10 or 8,9,10)")
    parser.add_argument("--rounds", default="16", help="Rounds range (e.g. 8-20:2)")
    parser.add_argument("--batch-sizes", default="256", help="Batch sizes (e.g. 128-256:16)")
    parser.add_argument("--runs", type=int, default=1, help="Runs per parameter combo")
    parser.add_argument("--seed", type=int, help="Seed for reproducible runs")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip correctness check vs reference",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Do not use tests/submission_tests.py (use internal runner instead)",
    )
    parser.add_argument("--output", default="cycle_sweep.csv", help="CSV output path")

    args = parser.parse_args()

    depths = parse_range_list(args.depths)
    rounds_list = parse_range_list(args.rounds)
    batch_sizes = parse_range_list(args.batch_sizes)

    results = []
    for forest_height in depths:
        for rounds in rounds_list:
            for batch_size in batch_sizes:
                for run in range(args.runs):
                    run_seed = None
                    if args.seed is not None:
                        run_seed = args.seed + run
                    try:
                        if args.kernel == "perf_takehome" and not args.standalone:
                            cycle = run_kernel_test_from_tests(
                                forest_height, rounds, batch_size, run_seed
                            )
                        else:
                            cycle = run_kernel_test(
                                args.kernel,
                                forest_height,
                                rounds,
                                batch_size,
                                run_seed,
                                not args.no_verify,
                            )
                        ok = True
                        error = ""
                    except Exception as exc:
                        cycle = None
                        ok = False
                        error = str(exc)
                    results.append(
                        {
                            "forest_height": forest_height,
                            "rounds": rounds,
                            "batch_size": batch_size,
                            "run": run,
                            "cycle": cycle,
                            "ok": ok,
                            "error": error,
                        }
                    )
                    status = "ok" if ok else "error"
                    print(
                        f"{status} depth={forest_height} rounds={rounds} batch={batch_size} "
                        f"run={run} cycle={cycle}"
                    )

    out_path = Path(args.output)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "forest_height",
                "rounds",
                "batch_size",
                "run",
                "cycle",
                "ok",
                "error",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary_rows = summarize(results)
    if summary_rows:
        print("\nSummary (per combo):")
        for row in summary_rows:
            print(
                "depth={forest_height} rounds={rounds} batch={batch_size} "
                "runs={runs} avg={avg_cycle:.2f} max={max_cycle} min={min_cycle}".format(
                    **row
                )
            )
        overall_cycles = [row["avg_cycle"] for row in summary_rows]
        overall_max = max(row["max_cycle"] for row in summary_rows)
        overall_avg = sum(overall_cycles) / len(overall_cycles)
        print(f"\nOverall avg of averages: {overall_avg:.2f}")
        print(f"Overall max cycle: {overall_max}")

    print(f"\nWrote CSV: {out_path}")


if __name__ == "__main__":
    main()
