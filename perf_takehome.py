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

    def _build_gather_instrs(self, vv, vi, k, v_nv, ta, fvp):
        """Build gather instruction list (ALU+LOAD only, no VALU).
        Returns list of instruction dicts for pipelined address computation + loads."""
        instrs = []
        # Build flat list of all load pairs
        all_loads = []
        for i in range(k):
            for j in range(0, VLEN, 2):
                all_loads.append((v_nv[i] + j, ta[i] + j,
                                  v_nv[i] + j + 1, ta[i] + j + 1))
        # First chunk ALU (standalone)
        instrs.append({"alu": [
            ("+", ta[0] + j, fvp, vi[0] + j) for j in range(VLEN)
        ]})
        # Subsequent chunks: overlap ALU with first load pair of prev chunk
        li = 0
        for i in range(1, k):
            lp = all_loads[li]
            instrs.append({
                "alu": [("+", ta[i] + j, fvp, vi[i] + j) for j in range(VLEN)],
                "load": [("load", lp[0], lp[1]), ("load", lp[2], lp[3])]
            })
            li += 1
        # Remaining loads
        for idx in range(li, len(all_loads)):
            lp = all_loads[idx]
            instrs.append({"load": [
                ("load", lp[0], lp[1]), ("load", lp[2], lp[3])
            ]})
        return instrs

    def _build_compute_instrs(self, vv, vi, k, v_nv, v_t1, v_t2, vhc, v_one, v_two, v_nn):
        """Build compute instruction list (VALU only).
        Returns list of 18 instruction dicts: XOR(1) + hash(12) + index(5)."""
        instrs = []
        # XOR
        instrs.append({"valu": [("^", vv[i], vv[i], v_nv[i]) for i in range(k)]})
        # Hash stages (interleaved across k chunks)
        for si, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            h1, h3 = vhc[si]
            ops = []
            for i in range(k):
                ops.extend([(op1, v_t1[i], vv[i], h1),
                            (op3, v_t2[i], vv[i], h3)])
            instrs.append({"valu": ops})
            instrs.append({"valu": [(op2, vv[i], v_t1[i], v_t2[i]) for i in range(k)]})
        # Index update (pure arithmetic)
        step1 = []
        for i in range(k):
            step1.extend([("&", v_t1[i], vv[i], v_one),
                          ("*", vi[i], vi[i], v_two)])
        instrs.append({"valu": step1})
        instrs.append({"valu": [("+", v_t1[i], v_t1[i], v_one) for i in range(k)]})
        instrs.append({"valu": [("+", vi[i], vi[i], v_t1[i]) for i in range(k)]})
        instrs.append({"valu": [("<", v_t1[i], vi[i], v_nn) for i in range(k)]})
        instrs.append({"valu": [("*", vi[i], vi[i], v_t1[i]) for i in range(k)]})
        return instrs

    def _emit_overlapped(self, compute_instrs, gather_instrs):
        """Merge compute (VALU) and gather (ALU+LOAD) instruction lists.
        No engine conflicts since they use completely separate engines."""
        n = max(len(compute_instrs), len(gather_instrs))
        for i in range(n):
            merged = {}
            if i < len(compute_instrs):
                merged.update(compute_instrs[i])
            if i < len(gather_instrs):
                merged.update(gather_instrs[i])
            self.instrs.append(merged)

    def build_kernel_optimized(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Vectorized kernel with software pipeline to hide load latency.

        Key optimizations:
        - 3-chunk interleaving fills 6/6 valu slots during hash ops
        - Arithmetic index update (no flow engine)
        - Inter-group pipeline: overlap next group's gather (LOAD+ALU) with
          current group's compute (VALU) since they use separate engines
        """
        # Scratch space addresses
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        tmp1 = self.alloc_scratch("tmp1")
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))
        self.add("flow", ("pause",))

        # 1. Bulk load values and indices into scratch
        bv, bi = self.bulk_load_into_scratch_space(batch_size)

        # 3 sets of vector temporaries for up to 3-chunk interleaving
        MAX_K = 3
        v_nv = [self.alloc_scratch(f"v_nv_{i}", VLEN) for i in range(MAX_K)]
        v_t1 = [self.alloc_scratch(f"v_t1_{i}", VLEN) for i in range(MAX_K)]
        v_t2 = [self.alloc_scratch(f"v_t2_{i}", VLEN) for i in range(MAX_K)]
        ta = [self.alloc_scratch(f"ta_{i}", VLEN) for i in range(MAX_K)]

        # Broadcast scalar constants to vectors
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_nn = self.alloc_scratch("v_nn", VLEN)

        self.add_multiple("valu", [
            ("vbroadcast", v_one, self.scratch_const(1)),
            ("vbroadcast", v_two, self.scratch_const(2)),
            ("vbroadcast", v_nn, self.scratch["n_nodes"]),
        ])

        # Pre-broadcast hash constants into vectors
        vhc = []
        for si, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            h1 = self.alloc_scratch(f"vh1_{si}", VLEN)
            h3 = self.alloc_scratch(f"vh3_{si}", VLEN)
            self.add_multiple("valu", [
                ("vbroadcast", h1, self.scratch_const(val1)),
                ("vbroadcast", h3, self.scratch_const(val3)),
            ])
            vhc.append((h1, h3))

        fvp = self.scratch["forest_values_p"]

        # Build group definitions: list of (vv, vi, k) for each group
        n_chunks = batch_size // VLEN
        groups = []
        c = 0
        while c < n_chunks:
            k = min(MAX_K, n_chunks - c)
            vv = [bv + (c + i) * VLEN for i in range(k)]
            vi = [bi + (c + i) * VLEN for i in range(k)]
            groups.append((vv, vi, k))
            c += k

        # Software pipeline: overlap next group's gather with current group's compute
        for r in range(rounds):
            # First group: standalone gather
            vv0, vi0, k0 = groups[0]
            gather_0 = self._build_gather_instrs(vv0, vi0, k0, v_nv, ta, fvp)
            self.instrs.extend(gather_0)

            for g in range(len(groups)):
                vv_g, vi_g, k_g = groups[g]
                compute = self._build_compute_instrs(
                    vv_g, vi_g, k_g, v_nv, v_t1, v_t2, vhc, v_one, v_two, v_nn)

                if g + 1 < len(groups):
                    # Overlap: current compute (VALU) + next gather (ALU+LOAD)
                    vv_next, vi_next, k_next = groups[g + 1]
                    next_gather = self._build_gather_instrs(
                        vv_next, vi_next, k_next, v_nv, ta, fvp)
                    self._emit_overlapped(compute, next_gather)
                else:
                    # Last group: standalone compute
                    self.instrs.extend(compute)

        # Write back results to memory
        self.bulk_store_to_memory(batch_size, bv, bi)

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
