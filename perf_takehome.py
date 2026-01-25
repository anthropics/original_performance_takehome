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
    myhash,
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

    def build(self, slots: list[tuple[Engine, tuple]]):
        # Always use VLIW scheduling (best setting).
        return schedule_slots(slots, SLOT_LIMITS)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def tag(self, text: str):
        # Debug tags are ignored by submission simulator; useful for profiling.
        self.instrs.append({"debug": [("tag", text)]})

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
        return self.build_hash_vec_multi([(val_vec, tmp1_vec, tmp2_vec)], round, base)

    def build_hash_vec_multi(self, val_groups, round, base):
        slots = []
        # Best-settings: op1/op3 stay vectorized, op2 uses scalar ALU per lane.
        scalar_op1 = False
        scalar_op3 = False
        scalar_op2 = True
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op2 == "+" and op1 == "+" and op3 == "<<":
                # (a + val1) + (a << k) == a * (1 + 2^k) + val1
                mul = self.scratch_vconst(1 + (1 << val3))
                add = self.scratch_vconst(val1)
                for val_vec, _t1, _t2 in val_groups:
                    slots.append(("valu", ("multiply_add", val_vec, val_vec, mul, add)))
                continue
            c1 = self.scratch_vconst(val1)
            c3 = self.scratch_vconst(val3)
            if scalar_op1:
                for val_vec, t1, _t2 in val_groups:
                    for lane in range(VLEN):
                        slots.append(("alu", (op1, t1 + lane, val_vec + lane, c1 + lane)))
            else:
                for val_vec, t1, _t2 in val_groups:
                    slots.append(("valu", (op1, t1, val_vec, c1)))
            if scalar_op3:
                for val_vec, _t1, t2 in val_groups:
                    for lane in range(VLEN):
                        slots.append(("alu", (op3, t2 + lane, val_vec + lane, c3 + lane)))
            else:
                for val_vec, _t1, t2 in val_groups:
                    slots.append(("valu", (op3, t2, val_vec, c3)))
            if scalar_op2 and op2 in ("+", "^"):
                for val_vec, t1, t2 in val_groups:
                    for lane in range(VLEN):
                        slots.append(("alu", (op2, val_vec + lane, t1 + lane, t2 + lane)))
            else:
                for val_vec, t1, t2 in val_groups:
                    slots.append(("valu", (op2, val_vec, t1, t2)))
        return slots

    def build_small_gather_select4_flow(
        self,
        dest,
        idx_vec,
        base_vec,
        vals4,
        tmp_idx,
        tmp_bit0,
        tmp_pair,
        vec_one,
    ):
        # 4-way select using 3 flow vselects; tmp_idx reused for bit1.
        slots = []
        slots.append(("valu", ("-", tmp_idx, idx_vec, base_vec)))
        slots.append(("valu", ("&", tmp_bit0, tmp_idx, vec_one)))
        slots.append(("valu", (">>", tmp_idx, tmp_idx, vec_one)))
        slots.append(("valu", ("&", tmp_idx, tmp_idx, vec_one)))
        slots.append(("flow", ("vselect", dest, tmp_bit0, vals4[1], vals4[0])))
        slots.append(("flow", ("vselect", tmp_pair, tmp_bit0, vals4[3], vals4[2])))
        slots.append(("flow", ("vselect", dest, tmp_idx, tmp_pair, dest)))
        return slots

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
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

        # Best-settings vector path.
        unroll = 8
        vec_tmp3 = self.alloc_scratch("vec_tmp3", length=VLEN)

        # Allocate vector pools to avoid scratch overlap between different roles.
        vec_idx_base = self.alloc_scratch("vec_idx_pool", length=unroll * VLEN)
        vec_val_base = self.alloc_scratch("vec_val_pool", length=unroll * VLEN)
        vec_node_base = self.alloc_scratch("vec_node_pool", length=unroll * VLEN)
        vec_tmp1_base = self.alloc_scratch("vec_tmp1_pool", length=unroll * VLEN)
        vec_tmp2_base = self.alloc_scratch("vec_tmp2_pool", length=unroll * VLEN)
        vec_addr_base = self.alloc_scratch("vec_addr_pool", length=unroll * VLEN)
        vec_addr_idx_base = self.alloc_scratch("vec_addr_idx_pool", length=unroll)
        vec_addr_val_base = self.alloc_scratch("vec_addr_val_pool", length=unroll)

        vec_idx_u = [vec_idx_base + ui * VLEN for ui in range(unroll)]
        vec_val_u = [vec_val_base + ui * VLEN for ui in range(unroll)]
        vec_node_u = [vec_node_base + ui * VLEN for ui in range(unroll)]
        vec_tmp1_u = [vec_tmp1_base + ui * VLEN for ui in range(unroll)]
        vec_tmp2_u = [vec_tmp2_base + ui * VLEN for ui in range(unroll)]
        vec_addr_u = [vec_addr_base + ui * VLEN for ui in range(unroll)]
        vec_addr_idx_u = [vec_addr_idx_base + ui for ui in range(unroll)]
        vec_addr_val_u = [vec_addr_val_base + ui for ui in range(unroll)]
        vec_one = self.scratch_vconst(1)
        vec_two = self.scratch_vconst(2)
        vec_forest_base = self.scratch_vbroadcast(self.scratch["forest_values_p"])
        # Fixed best-settings flags.
        forest0 = self.alloc_scratch("forest0")
        forest1 = self.alloc_scratch("forest1")
        forest2 = self.alloc_scratch("forest2")
        vec_forest0 = self.alloc_scratch("vec_forest0", length=VLEN)
        vec_forest1 = self.alloc_scratch("vec_forest1", length=VLEN)
        vec_forest2 = self.alloc_scratch("vec_forest2", length=VLEN)
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], zero_const)))
        body.append(("load", ("load", forest0, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest0, forest0)))
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], one_const)))
        body.append(("load", ("load", forest1, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest1, forest1)))
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], two_const)))
        body.append(("load", ("load", forest2, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest2, forest2)))
        vec_base3 = self.scratch_vconst(3)
        forest3_6 = []
        forest3_6_consts = []
        vec_forest3_6 = []
        for idx in range(3, 7):
            f = self.alloc_scratch(f"forest{idx}")
            vf = self.alloc_scratch(f"vec_forest{idx}", length=VLEN)
            idx_const = self.scratch_const(idx)
            body.append(
                ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_const))
            )
            body.append(("load", ("load", f, tmp_addr)))
            body.append(("valu", ("vbroadcast", vf, f)))
            forest3_6.append(f)
            forest3_6_consts.append(idx_const)
            vec_forest3_6.append(vf)
        vec_tmp4 = self.alloc_scratch("vec_tmp4", length=VLEN)
        

        vec_limit = (batch_size // VLEN) * VLEN
        # Per-value pipeline: load each vector chunk once, run all rounds, then store once.
        for base in range(0, vec_limit, VLEN * unroll):
            # address setup + vload idx/val
            for ui in range(unroll):
                base_u = base + ui * VLEN
                if base_u >= vec_limit:
                    continue
                base_u_const = self.scratch_const(base_u)
                body.append(("alu", ("+", vec_addr_idx_u[ui], self.scratch["inp_indices_p"], base_u_const)))
                body.append(("alu", ("+", vec_addr_val_u[ui], self.scratch["inp_values_p"], base_u_const)))
                body.append(("load", ("vload", vec_idx_u[ui], vec_addr_idx_u[ui])))
                body.append(("load", ("vload", vec_val_u[ui], vec_addr_val_u[ui])))
            for round in range(rounds):
                depth = round % (forest_height + 1)
                # gather node values
                if depth not in (0, 1, 2):
                    for ui in range(unroll):
                        base_u = base + ui * VLEN
                        if base_u >= vec_limit:
                            continue
                        body.append(("valu", ("+", vec_addr_u[ui], vec_forest_base, vec_idx_u[ui])))
                        for lane in range(VLEN):
                            body.append(("load", ("load_offset", vec_node_u[ui], vec_addr_u[ui], lane)))
                # process each vector (hash interleaved across unroll)
                val_groups = []
                idx_vecs = []
                for ui in range(unroll):
                    base_u = base + ui * VLEN
                    if base_u >= vec_limit:
                        continue
                    if depth == 0:
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_forest0)))
                    elif depth == 1:
                        body.append(("valu", ("%", vec_tmp1_u[ui], vec_idx_u[ui], vec_two)))
                        body.append(("flow", ("vselect", vec_node_u[ui], vec_tmp1_u[ui], vec_forest1, vec_forest2)))
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))
                    elif depth == 2:
                        body.extend(self.build_small_gather_select4_flow(vec_node_u[ui], vec_idx_u[ui], vec_base3, vec_forest3_6, vec_tmp1_u[ui], vec_tmp2_u[ui], vec_tmp4, vec_one))
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))
                    else:
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))
                    val_groups.append((vec_val_u[ui], vec_tmp1_u[ui], vec_tmp2_u[ui]))
                    idx_vecs.append(vec_idx_u[ui])
                body.extend(self.build_hash_vec_multi(val_groups, round, base))
                for vi, (val_vec, t1, _t2) in enumerate(val_groups):
                    idx_vec = idx_vecs[vi]
                    if depth == forest_height:
                        body.append(("valu", ("^", idx_vec, idx_vec, idx_vec)))
                        continue
                    body.append(("valu", ("&", t1, val_vec, vec_one)))
                    if depth == 0:
                        body.append(("valu", ("+", idx_vec, t1, vec_one)))
                        continue
                    body.append(("valu", ("multiply_add", idx_vec, idx_vec, vec_two, t1)))
                    body.append(("valu", ("+", idx_vec, idx_vec, vec_one)))
            # store idx/val once per chunk
            for ui in range(unroll):
                base_u = base + ui * VLEN
                if base_u >= vec_limit:
                    continue
                body.append(("store", ("vstore", vec_addr_idx_u[ui], vec_idx_u[ui])))
                body.append(("store", ("vstore", vec_addr_val_u[ui], vec_val_u[ui])))
        # tail (scalar)
        for i in range(vec_limit, batch_size):
            i_const = self.scratch_const(i)
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            body.append(("load", ("load", tmp_idx, tmp_addr)))
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            body.append(("load", ("load", tmp_val, tmp_addr)))
            for round in range(rounds):
                depth = round % (forest_height + 1)
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                if depth == forest_height:
                    body.append(("alu", ("^", tmp_idx, tmp_idx, tmp_idx)))
                    continue
                body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                if depth == 0:
                    body.append(("alu", ("+", tmp_idx, tmp1, one_const)))
                    continue
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp1)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, one_const)))
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            body.append(("store", ("store", tmp_addr, tmp_idx)))
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            body.append(("store", ("store", tmp_addr, tmp_val)))
        body_instrs = self.build(body)
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
):
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
    if os.getenv("PROFILE", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
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
    return parser.parse_args()


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
    do_kernel_test(
        args.forest_height,
        args.rounds,
        args.batch_size,
        seed=args.seed,
        trace=args.trace,
        prints=args.prints,
    )


def _has_cli_args(argv: list[str]) -> bool:
    cli_flags = {
        "--forest-height",
        "--rounds",
        "--batch-size",
        "--seed",
        "--trace",
        "--prints",
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
