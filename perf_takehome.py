"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import argparse
import os
import sys
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)
from vliw_scheduler import schedule_slots


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}
        self.vec_broadcast_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        if _env_flag("VLIW"):
            vliw = True
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        return schedule_slots(slots, SLOT_LIMITS)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vec_const_map:
            scalar = self.scratch_const(val, name=None)
            addr = self.alloc_scratch(name, length=VLEN)
            self.add("valu", ("vbroadcast", addr, scalar))
            self.vec_const_map[val] = addr
        return self.vec_const_map[val]

    def scratch_vbroadcast(self, scalar_addr, name=None):
        if scalar_addr not in self.vec_broadcast_map:
            addr = self.alloc_scratch(name, length=VLEN)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vec_broadcast_map[scalar_addr] = addr
        return self.vec_broadcast_map[scalar_addr]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vec(self, val_vec, tmp1_vec, tmp2_vec, round, base):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op2 == "+" and op1 == "+" and op3 == "<<":
                # (a + val1) + (a << k) == a * (1 + 2^k) + val1
                mul = self.scratch_vconst(1 + (1 << val3))
                add = self.scratch_vconst(val1)
                slots.append(("valu", ("multiply_add", val_vec, val_vec, mul, add)))
                continue
            c1 = self.scratch_vconst(val1)
            c3 = self.scratch_vconst(val3)
            slots.append(("valu", (op1, tmp1_vec, val_vec, c1)))
            slots.append(("valu", (op3, tmp2_vec, val_vec, c3)))
            slots.append(("valu", (op2, val_vec, tmp1_vec, tmp2_vec)))
        return slots

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        *,
        vliw: bool = False,
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers (used in both paths)
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        if _env_flag("VEC"):
            # Unroll factor for vector chunks (1, 2, 4, 8)
            if _env_flag("VEC2"):
                unroll = 2
            elif _env_flag("VEC4"):
                unroll = 4
            else:
                try:
                    unroll = int(os.getenv("VEC_UNROLL", "1"))
                except ValueError:
                    unroll = 1
            if unroll not in (1, 2, 4, 8):
                unroll = 1

            vec_tmp1 = self.alloc_scratch("vec_tmp1", length=VLEN)
            vec_tmp2 = self.alloc_scratch("vec_tmp2", length=VLEN)
            vec_tmp3 = self.alloc_scratch("vec_tmp3", length=VLEN)

            vec_idx_u = []
            vec_val_u = []
            vec_node_u = []
            vec_addr_u = []
            vec_addr_idx_u = []
            vec_addr_val_u = []
            for ui in range(unroll):
                vec_idx_u.append(self.alloc_scratch(f"vec_idx_{ui}", length=VLEN))
                vec_val_u.append(self.alloc_scratch(f"vec_val_{ui}", length=VLEN))
                vec_node_u.append(self.alloc_scratch(f"vec_node_{ui}", length=VLEN))
                vec_addr_u.append(self.alloc_scratch(f"vec_addr_{ui}", length=VLEN))
                vec_addr_idx_u.append(self.alloc_scratch(f"vec_addr_idx_{ui}"))
                vec_addr_val_u.append(self.alloc_scratch(f"vec_addr_val_{ui}"))
            vec_zero = self.scratch_vconst(0)
            vec_one = self.scratch_vconst(1)
            vec_two = self.scratch_vconst(2)
            vec_n_nodes = self.scratch_vbroadcast(self.scratch["n_nodes"])
            vec_forest_base = self.scratch_vbroadcast(self.scratch["forest_values_p"])

            for round in range(rounds):
                vec_limit = (batch_size // VLEN) * VLEN
                if unroll > 0:
                    for base in range(0, vec_limit, VLEN * unroll):
                        # address setup
                        for ui in range(unroll):
                            base_u = base + ui * VLEN
                            if base_u >= vec_limit:
                                continue
                            base_u_const = self.scratch_const(base_u)
                            body.append(
                                (
                                    "alu",
                                    (
                                        "+",
                                        vec_addr_idx_u[ui],
                                        self.scratch["inp_indices_p"],
                                        base_u_const,
                                    ),
                                )
                            )
                            body.append(
                                (
                                    "alu",
                                    (
                                        "+",
                                        vec_addr_val_u[ui],
                                        self.scratch["inp_values_p"],
                                        base_u_const,
                                    ),
                                )
                            )

                        # vload idx/val
                        for ui in range(unroll):
                            base_u = base + ui * VLEN
                            if base_u >= vec_limit:
                                continue
                            body.append(("load", ("vload", vec_idx_u[ui], vec_addr_idx_u[ui])))
                            body.append(("load", ("vload", vec_val_u[ui], vec_addr_val_u[ui])))

                        # gather node values
                        for ui in range(unroll):
                            base_u = base + ui * VLEN
                            if base_u >= vec_limit:
                                continue
                            body.append(
                                ("valu", ("+", vec_addr_u[ui], vec_forest_base, vec_idx_u[ui]))
                            )
                            for lane in range(VLEN):
                                body.append(
                                    ("load", ("load_offset", vec_node_u[ui], vec_addr_u[ui], lane))
                                )

                        # process each vector
                        for ui in range(unroll):
                            base_u = base + ui * VLEN
                            if base_u >= vec_limit:
                                continue
                            body.append(
                                ("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui]))
                            )
                            body.extend(
                                self.build_hash_vec(
                                    vec_val_u[ui], vec_tmp1, vec_tmp2, round, base_u
                                )
                            )
                            body.append(("valu", ("%", vec_tmp1, vec_val_u[ui], vec_two)))
                            body.append(("valu", ("==", vec_tmp1, vec_tmp1, vec_zero)))
                            body.append(
                                ("flow", ("vselect", vec_tmp3, vec_tmp1, vec_one, vec_two))
                            )
                            body.append(("valu", ("*", vec_idx_u[ui], vec_idx_u[ui], vec_two)))
                            body.append(("valu", ("+", vec_idx_u[ui], vec_idx_u[ui], vec_tmp3)))
                            body.append(("valu", ("<", vec_tmp1, vec_idx_u[ui], vec_n_nodes)))
                            body.append(
                                ("flow", ("vselect", vec_idx_u[ui], vec_tmp1, vec_idx_u[ui], vec_zero))
                            )
                            body.append(("store", ("vstore", vec_addr_idx_u[ui], vec_idx_u[ui])))
                            body.append(("store", ("vstore", vec_addr_val_u[ui], vec_val_u[ui])))
                # tail (scalar)
                for i in range(vec_limit, batch_size):
                    i_const = self.scratch_const(i)
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                    body.append(("load", ("load", tmp_idx, tmp_addr)))
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                    body.append(("load", ("load", tmp_val, tmp_addr)))
                    body.append(
                        ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                    )
                    body.append(("load", ("load", tmp_node_val, tmp_addr)))
                    body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                    body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                    body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                    body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                    body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                    body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                    body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                    body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                    body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                    body.append(("store", ("store", tmp_addr, tmp_idx)))
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                    body.append(("store", ("store", tmp_addr, tmp_val)))
        else:
            for round in range(rounds):
                for i in range(batch_size):
                    i_const = self.scratch_const(i)
                    # idx = mem[inp_indices_p + i]
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                    body.append(("load", ("load", tmp_idx, tmp_addr)))
                    body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                    # val = mem[inp_values_p + i]
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                    body.append(("load", ("load", tmp_val, tmp_addr)))
                    body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                    # node_val = mem[forest_values_p + idx]
                    body.append(
                        ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                    )
                    body.append(("load", ("load", tmp_node_val, tmp_addr)))
                    body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                    # val = myhash(val ^ node_val)
                    body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                    body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                    body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                    body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                    body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                    body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                    body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                    body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                    # idx = 0 if idx >= n_nodes else idx
                    body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                    body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                    body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                    # mem[inp_indices_p + i] = idx
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                    body.append(("store", ("store", tmp_addr, tmp_idx)))
                    # mem[inp_values_p + i] = val
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                    body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body, vliw=vliw)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    vliw: bool = False,
):
    if _env_flag("VLIW"):
        vliw = True
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(
        forest.height,
        len(forest.values),
        len(inp.indices),
        rounds,
        vliw=vliw,
    )
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    if _env_flag("PROFILE"):
        machine.profile = True
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


def _parse_args():
    parser = argparse.ArgumentParser(description="Run kernel test.")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--prints", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline and VLIW back-to-back with the same parameters.",
    )
    return parser.parse_args()


def _env_flag(name: str) -> bool:
    val = os.getenv(name, "").strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


def _main():
    args = _parse_args()
    vliw = _env_flag("VLIW")
    if args.compare:
        print("== Baseline ==")
        do_kernel_test(
            args.forest_height,
            args.rounds,
            args.batch_size,
            seed=args.seed,
            trace=args.trace,
            prints=args.prints,
            vliw=False,
        )
        print("== VLIW ==")
        do_kernel_test(
            args.forest_height,
            args.rounds,
            args.batch_size,
            seed=args.seed,
            trace=args.trace,
            prints=args.prints,
            vliw=True,
        )
    else:
        do_kernel_test(
            args.forest_height,
            args.rounds,
            args.batch_size,
            seed=args.seed,
            trace=args.trace,
            prints=args.prints,
            vliw=vliw,
        )


def _has_cli_args(argv: list[str]) -> bool:
    cli_flags = {
        "--forest-height",
        "--rounds",
        "--batch-size",
        "--seed",
        "--trace",
        "--prints",
        "--compare",
    }
    return any(arg in cli_flags for arg in argv[1:])


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    if _has_cli_args(sys.argv):
        _main()
    else:
        unittest.main()
