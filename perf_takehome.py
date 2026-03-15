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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_multiple(self, engine, slot):
        self.instrs.append({ engine: slot })

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            self.add_multiple("alu", [
                (op1, tmp1, val_hash_addr, self.scratch_const(val1)),
                (op3, tmp2, val_hash_addr, self.scratch_const(val3)),
            ])
            self.add("alu", (op2, val_hash_addr, tmp1, tmp2))
            self.add("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))

    def bulk_load_into_scratch_space(self, batch_size: int):
        """Load batch values and indices from memory into contiguous scratch space.
        Uses pipelining: compute next addresses while loading current chunk."""
        batch_offset_values = self.alloc_scratch("batch_offset_values", batch_size)
        batch_offset_indices = self.alloc_scratch("batch_offset_indices", batch_size)
        index_into_memory1 = self.alloc_scratch("index_into_memory1")
        index_into_memory2 = self.alloc_scratch("index_into_memory2")
        n_chunks = batch_size // VLEN
        for s in range(n_chunks):
            memory_offset = self.scratch_const(s * VLEN)
            if s == 0:
                # First iteration: just compute addresses (no data to load yet)
                self.add_multiple("alu", [
                    ("+", index_into_memory1, self.scratch["inp_values_p"], memory_offset),
                    ("+", index_into_memory2, self.scratch["inp_indices_p"], memory_offset)
                ])
            else:
                # Pipeline: load previous chunk while computing next addresses
                self.instrs.append({
                    "alu": [
                        ("+", index_into_memory1, self.scratch["inp_values_p"], memory_offset),
                        ("+", index_into_memory2, self.scratch["inp_indices_p"], memory_offset)
                    ],
                    "load": [
                        ("vload", batch_offset_values + (s - 1) * VLEN, index_into_memory1),
                        ("vload", batch_offset_indices + (s - 1) * VLEN, index_into_memory2)
                    ]
                })
        # Final load (no more addresses to compute)
        self.add_multiple("load", [
            ("vload", batch_offset_values + (n_chunks - 1) * VLEN, index_into_memory1),
            ("vload", batch_offset_indices + (n_chunks - 1) * VLEN, index_into_memory2)
        ])
        return batch_offset_values, batch_offset_indices

    def bulk_store_to_memory(self, batch_size: int, batch_values_offset: int, batch_indices_offset: int):
        """Store batch values and indices from scratch space back to memory.
        Uses pipelining: compute next addresses while storing current chunk."""
        store_addr1 = self.alloc_scratch("store_addr1")
        store_addr2 = self.alloc_scratch("store_addr2")
        n_chunks = batch_size // VLEN
        for s in range(n_chunks):
            memory_offset = self.scratch_const(s * VLEN)
            if s == 0:
                self.add_multiple("alu", [
                    ("+", store_addr1, self.scratch["inp_values_p"], memory_offset),
                    ("+", store_addr2, self.scratch["inp_indices_p"], memory_offset)
                ])
            else:
                self.instrs.append({
                    "alu": [
                        ("+", store_addr1, self.scratch["inp_values_p"], memory_offset),
                        ("+", store_addr2, self.scratch["inp_indices_p"], memory_offset)
                    ],
                    "store": [
                        ("vstore", store_addr1, batch_values_offset + (s - 1) * VLEN),
                        ("vstore", store_addr2, batch_indices_offset + (s - 1) * VLEN)
                    ]
                })
        # Final store
        self.add_multiple("store", [
            ("vstore", store_addr1, batch_values_offset + (n_chunks - 1) * VLEN),
            ("vstore", store_addr2, batch_indices_offset + (n_chunks - 1) * VLEN)
        ])

    def build_kernel_optimized(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Vectorized kernel using valu and v* operations.

        1. bulk load entire batch from memory into scratch space
        for each chunk of VLEN elements:
          for each round:
            2. gather node values (scalar loads since indices are non-contiguous)
            3. use valu to compute XOR and iterate through hash stages
            4. use valu/vselect for index update
        5. write back results into memory
        """
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
        tmp1 = self.alloc_scratch("tmp1")
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))
        self.add("flow", ("pause",))

        # 1. Bulk load values and indices into scratch
        batch_values_offset, batch_indices_offset = self.bulk_load_into_scratch_space(batch_size)

        # Allocate vector working space
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        tmp_addrs = self.alloc_scratch("tmp_addrs", VLEN)

        # Broadcast scalar constants to vectors
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        zero_s = self.scratch_const(0)
        one_s = self.scratch_const(1)
        two_s = self.scratch_const(2)

        self.add_multiple("valu", [
            ("vbroadcast", v_zero, zero_s),
            ("vbroadcast", v_one, one_s),
            ("vbroadcast", v_two, two_s),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ])

        # Pre-broadcast hash constants into vectors
        v_hash_consts = []
        for si, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_val1 = self.alloc_scratch(f"vh1_{si}", VLEN)
            v_val3 = self.alloc_scratch(f"vh3_{si}", VLEN)
            self.add_multiple("valu", [
                ("vbroadcast", v_val1, self.scratch_const(val1)),
                ("vbroadcast", v_val3, self.scratch_const(val3)),
            ])
            v_hash_consts.append((v_val1, v_val3))

        # Main computation loop
        n_chunks = batch_size // VLEN
        for c in range(n_chunks):
            v_val = batch_values_offset + c * VLEN
            v_idx = batch_indices_offset + c * VLEN

            for r in range(rounds):
                # Gather node values: compute addr[j] = forest_values_p + idx[j]
                self.add_multiple("alu", [
                    ("+", tmp_addrs + j, self.scratch["forest_values_p"], v_idx + j)
                    for j in range(VLEN)
                ])
                # Scalar loads for non-contiguous node values (2 per cycle)
                for j in range(0, VLEN, 2):
                    self.add_multiple("load", [
                        ("load", v_node_val + j, tmp_addrs + j),
                        ("load", v_node_val + j + 1, tmp_addrs + j + 1),
                    ])

                # XOR val with node_val
                self.add("valu", ("^", v_val, v_val, v_node_val))

                # Hash stages
                for si, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    vh1, vh3 = v_hash_consts[si]
                    self.add_multiple("valu", [
                        (op1, v_tmp1, v_val, vh1),
                        (op3, v_tmp2, v_val, vh3),
                    ])
                    self.add("valu", (op2, v_val, v_tmp1, v_tmp2))

                # Index update: idx = 2*idx + (1 if val%2==0 else 2)
                self.add("valu", ("%", v_tmp1, v_val, v_two))
                self.add("valu", ("==", v_tmp1, v_tmp1, v_zero))
                # Overlap vselect (flow) with multiply (valu) - independent operations
                self.instrs.append({
                    "flow": [("vselect", v_tmp3, v_tmp1, v_one, v_two)],
                    "valu": [("*", v_idx, v_idx, v_two)],
                })
                self.add("valu", ("+", v_idx, v_idx, v_tmp3))
                # Wrap: idx = 0 if idx >= n_nodes else idx
                self.add("valu", ("<", v_tmp1, v_idx, v_n_nodes))
                self.add("flow", ("vselect", v_idx, v_tmp1, v_idx, v_zero))

        # 5. Write back results to memory
        self.bulk_store_to_memory(batch_size, batch_values_offset, batch_indices_offset)

        self.instrs.append({"flow": [("pause",)]})



    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        self.build_kernel_optimized(forest_height, n_nodes, batch_size, rounds)

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
