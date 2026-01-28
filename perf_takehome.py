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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

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
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, length=VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vhash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # Use multiply_add
                factor = 1 + (1 << val3)
                slots.append(("valu", ("multiply_add", val_hash_addr, val_hash_addr, self.scratch_vconst(factor), self.scratch_vconst(val1))))
            else:
                slots.append(("valu", (op1, tmp1, val_hash_addr, self.scratch_vconst(val1))))
                slots.append(("valu", (op3, tmp2, val_hash_addr, self.scratch_vconst(val3))))
                slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("vcompare", val_hash_addr, [(round, i+idx, "hash_stage", hi) for idx in range(VLEN)])))
        
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation using SIMD for batch dimension.
        Assume batch_size is a multiple of VLEN.
        """
        num_blocks = batch_size // VLEN

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

        # Precompute the idx: inp_indices_p + i and inp_values_p + i
        idx_ptrs = self.alloc_scratch("idx_ptrs", batch_size)
        val_ptrs = self.alloc_scratch("val_ptrs", batch_size)
        self.add("alu", ("+", idx_ptrs, self.scratch["inp_indices_p"], zero_const))
        self.add("alu", ("+", val_ptrs, self.scratch["inp_values_p"], zero_const))
        for i in range(1, batch_size):
            self.add("alu", ("+", idx_ptrs + i, idx_ptrs + i - 1, one_const))
            self.add("alu", ("+", val_ptrs + i, val_ptrs + i - 1, one_const))
        
        # Vector scratch registers and broadcasted constants.
        vtmp1 = self.alloc_scratch("vtmp1", VLEN)
        vtmp2 = self.alloc_scratch("vtmp2", VLEN)
        vtmp3 = self.alloc_scratch("vtmp3", VLEN)

        vone_const = self.scratch_vconst(1, "vone")
        vtwo_const = self.scratch_vconst(2, "vtwo")

        vtmp_idx = self.alloc_scratch("vtmp_idx", VLEN)
        vtmp_val = self.alloc_scratch("vtmp_val", VLEN)
        vtmp_node_val = self.alloc_scratch("vtmp_node_val", VLEN)
        vtmp_addr = self.alloc_scratch("vtmp_addr", VLEN)
        
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        v_forest_p = self.scratch_vconst(self.scratch["forest_values_p"])

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                # idx = mem[inp_indices_p + i]
                idx_ptr = idx_ptrs + i
                body.append(("load", ("vload", vtmp_idx, idx_ptr)))
                body.append(("debug", ("vcompare", vtmp_idx, [(round, i+j, "idx") for j in range(VLEN)])))
                # val = mem[inp_values_p + i]
                val_ptr = val_ptrs + i
                body.append(("load", ("vload", vtmp_val, val_ptr)))
                body.append(("debug", ("vcompare", vtmp_val, [(round, i+j, "val") for j in range(VLEN)])))
                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", vtmp_addr, v_forest_p, vtmp_idx)))
                for j in range(VLEN):
                    body.append(("load", ("load_offset", vtmp_node_val, vtmp_addr, j)))
                body.append(("debug", ("vcompare", vtmp_node_val, [(round, i+j, "node_val") for j in range(VLEN)])))
                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", vtmp_val, vtmp_val, vtmp_node_val)))
                body.extend(self.build_vhash(vtmp_val, vtmp1, vtmp2, round, i))
                body.append(("debug", ("vcompare", vtmp_val, [(round, i+j, "hashed_val") for j in range(VLEN)])))
                # Change to idx = 2*idx + 1 + (val % 2)
                body.append(("valu", ("&", vtmp3, vtmp_val, vone_const)))
                body.append(("valu", ("+", vtmp3, vtmp3, vone_const)))
                body.append(("valu", ("multiply_add", vtmp_idx, vtmp_idx, vtwo_const, vtmp3)))
                body.append(("debug", ("vcompare", vtmp_idx, [(round, i+j, "next_idx") for j in range(VLEN)])))
                # Change to idx = idx * (idx < n_nodes)
                body.append(("valu", ("<", vtmp1, vtmp_idx, v_n_nodes)))
                body.append(("valu", ("*", vtmp_idx, vtmp1, vtmp_idx)))
                body.append(("debug", ("vcompare", vtmp_idx, [(round, i+j, "wrapped_idx") for j in range(VLEN)])))
                # mem[inp_indices_p + i] = idx
                body.append(("store", ("vstore", idx_ptr, vtmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("store", ("vstore", val_ptr, vtmp_val)))

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
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
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
    unittest.main()
