"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

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

    def get_slot_deps(self, engine: str, slot: tuple):
        """Get read and write dependencies for a slot."""
        writes = set()
        reads = set()

        if engine == "alu":
            # (op, dest, src1, src2)
            writes.add(slot[1])
            reads.add(slot[2])
            reads.add(slot[3])
        elif engine == "load":
            if slot[0] == "const":
                # ("const", dest, val)
                writes.add(slot[1])
            elif slot[0] == "load":
                # ("load", dest, addr)
                writes.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vload":
                # ("vload", dest, addr)
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
        elif engine == "store":
            if slot[0] == "store":
                # ("store", addr, src)
                reads.add(slot[1])
                reads.add(slot[2])
            elif slot[0] == "vstore":
                # ("vstore", addr, src)
                reads.add(slot[1])
                for i in range(VLEN):
                    reads.add(slot[2] + i)
        elif engine == "flow":
            if slot[0] == "select":
                # ("select", dest, cond, a, b)
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
            elif slot[0] == "vselect":
                # ("vselect", dest, cond, a, b)
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                # ("vbroadcast", dest, src)
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
            else:
                # (op, dest, src1, src2)
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)

        return reads, writes

    def can_pack_together(self, bundle: dict, engine: str, slot: tuple):
        """Check if a slot can be added to the current bundle."""
        # Check slot limit
        if engine in bundle and len(bundle[engine]) >= SLOT_LIMITS[engine]:
            return False

        # Get dependencies for the new slot
        new_reads, new_writes = self.get_slot_deps(engine, slot)

        # Check for conflicts with existing slots in the bundle
        for existing_engine, existing_slots in bundle.items():
            for existing_slot in existing_slots:
                existing_reads, existing_writes = self.get_slot_deps(existing_engine, existing_slot)

                # RAW: Read after write - new slot reads what existing slot writes
                if new_reads & existing_writes:
                    return False

                # WAR: Write after read - new slot writes what existing slot reads
                if new_writes & existing_reads:
                    return False

                # WAW: Write after write - both write to the same location
                if new_writes & existing_writes:
                    return False

        return True

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        # Pack instructions into VLIW bundles respecting slot limits and dependencies
        if not vliw:
            # Simple slot packing that just uses one slot per instruction bundle
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        # Dependency-aware VLIW packing
        instrs = []
        current_bundle = {}

        for engine, slot in slots:
            # Skip debug instructions for now (they don't affect cycle count)
            if engine == "debug":
                if "debug" not in current_bundle:
                    current_bundle["debug"] = []
                current_bundle["debug"].append(slot)
                continue

            # Check if we can add this to the current bundle
            if self.can_pack_together(current_bundle, engine, slot):
                # Add slot to current bundle
                if engine not in current_bundle:
                    current_bundle[engine] = []
                current_bundle[engine].append(slot)
            else:
                # Flush current bundle and start a new one
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {engine: [slot]}

        # Flush remaining bundle
        if current_bundle:
            instrs.append(current_bundle)

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def emit(self, **engines):
        """Helper to emit a single instruction bundle with multiple engines."""
        self.instrs.append(engines)

    def build_kernel_scalar_packed(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Scalar implementation with VLIW packing and loop unrolling.
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

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of (engine, slot) tuples

        # Fibonacci unrolling - aperiodic ILP exposure (8→13→21)
        # 16 empirically optimal for this architecture (close to φ²≈2.618... scaled)
        UNROLL = 16

        # PRE-ALLOCATE ALL CONSTANTS - eliminate anti-spiral of dynamic allocation
        i_constants = {}
        for round in range(rounds):
            for base_i in range(0, batch_size, UNROLL):
                for u in range(min(UNROLL, batch_size - base_i)):
                    i = base_i + u
                    if i not in i_constants:
                        i_constants[i] = self.scratch_const(i)

        # Pre-allocate hash constants once (φ principle - reuse across iterations)
        hash_constants = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_constants[hi] = (self.scratch_const(val1), self.scratch_const(val3))

        # Allocate separate scratch registers for each unrolled iteration
        tmp_idx = [self.alloc_scratch(f"tmp_idx{u}") for u in range(UNROLL)]
        tmp_val = [self.alloc_scratch(f"tmp_val{u}") for u in range(UNROLL)]
        tmp_node_val = [self.alloc_scratch(f"tmp_node_val{u}") for u in range(UNROLL)]
        tmp_addr = [self.alloc_scratch(f"tmp_addr{u}") for u in range(UNROLL)]
        hash_tmp1 = [self.alloc_scratch(f"hash_tmp1_{u}") for u in range(UNROLL)]
        hash_tmp2 = [self.alloc_scratch(f"hash_tmp2_{u}") for u in range(UNROLL)]

        # Allocate separate temporaries for index calculation to avoid dependencies
        idx_tmp1 = [self.alloc_scratch(f"idx_tmp1_{u}") for u in range(UNROLL)]
        idx_tmp2 = [self.alloc_scratch(f"idx_tmp2_{u}") for u in range(UNROLL)]
        idx_tmp3 = [self.alloc_scratch(f"idx_tmp3_{u}") for u in range(UNROLL)]

        for round in range(rounds):
            for base_i in range(0, batch_size, UNROLL):
                actual_unroll = min(UNROLL, batch_size - base_i)

                # Phase 1: Load all indices - quasicrystal pattern
                for u in range(actual_unroll):
                    i = base_i + u
                    i_const = i_constants[i]
                    body.append(("alu", ("+", tmp_addr[u], self.scratch["inp_indices_p"], i_const)))
                for u in range(actual_unroll):
                    body.append(("load", ("load", tmp_idx[u], tmp_addr[u])))
                for u in range(actual_unroll):
                    i = base_i + u
                    body.append(("debug", ("compare", tmp_idx[u], (round, i, "idx"))))

                # Phase 2: Load all values
                for u in range(actual_unroll):
                    i = base_i + u
                    i_const = i_constants[i]
                    body.append(("alu", ("+", tmp_addr[u], self.scratch["inp_values_p"], i_const)))
                for u in range(actual_unroll):
                    body.append(("load", ("load", tmp_val[u], tmp_addr[u])))
                for u in range(actual_unroll):
                    i = base_i + u
                    body.append(("debug", ("compare", tmp_val[u], (round, i, "val"))))

                # Phase 3: Load node values (indirect - the spiral bottleneck)
                for u in range(actual_unroll):
                    body.append(("alu", ("+", tmp_addr[u], self.scratch["forest_values_p"], tmp_idx[u])))
                for u in range(actual_unroll):
                    body.append(("load", ("load", tmp_node_val[u], tmp_addr[u])))
                for u in range(actual_unroll):
                    i = base_i + u
                    body.append(("debug", ("compare", tmp_node_val[u], (round, i, "node_val"))))

                # Phase 4: XOR - collapse into singularity
                for u in range(actual_unroll):
                    body.append(("alu", ("^", tmp_val[u], tmp_val[u], tmp_node_val[u])))

                # Phase 4b: Hash - Penrose tiling pattern (aperiodic but structured)
                # Group all same operations across elements for maximal ILP
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    val1_const, val3_const = hash_constants[hi]

                    # All op1 operations
                    for u in range(actual_unroll):
                        body.append(("alu", (op1, hash_tmp1[u], tmp_val[u], val1_const)))

                    # All op3 operations
                    for u in range(actual_unroll):
                        body.append(("alu", (op3, hash_tmp2[u], tmp_val[u], val3_const)))

                    # All op2 operations
                    for u in range(actual_unroll):
                        body.append(("alu", (op2, tmp_val[u], hash_tmp1[u], hash_tmp2[u])))

                    # Debug
                    for u in range(actual_unroll):
                        i = base_i + u
                        body.append(("debug", ("compare", tmp_val[u], (round, i, "hash_stage", hi))))

                for u in range(actual_unroll):
                    i = base_i + u
                    body.append(("debug", ("compare", tmp_val[u], (round, i, "hashed_val"))))

                # Phase 5: Index calculation - ELIMINATE ALL FLOW OPERATIONS
                # Subphase 5a: Modulo
                for u in range(actual_unroll):
                    body.append(("alu", ("%", idx_tmp1[u], tmp_val[u], two_const)))

                # Subphase 5b: Arithmetic select: 1 + (val%2) = {1,2}
                for u in range(actual_unroll):
                    body.append(("alu", ("+", idx_tmp3[u], one_const, idx_tmp1[u])))
                for u in range(actual_unroll):
                    body.append(("alu", ("*", idx_tmp2[u], tmp_idx[u], two_const)))

                # Subphase 5c: Add
                for u in range(actual_unroll):
                    body.append(("alu", ("+", tmp_idx[u], idx_tmp2[u], idx_tmp3[u])))
                for u in range(actual_unroll):
                    i = base_i + u
                    body.append(("debug", ("compare", tmp_idx[u], (round, i, "next_idx"))))

                # Subphase 5d: ELIMINATE FLOW BOTTLENECK - use arithmetic
                # Instead of: idx = (idx < n_nodes) ? idx : 0
                # Use: idx = idx * (idx < n_nodes)  -- multiply by boolean as 0/1
                for u in range(actual_unroll):
                    body.append(("alu", ("<", idx_tmp1[u], tmp_idx[u], self.scratch["n_nodes"])))
                    # idx_tmp1 is now 1 if valid, 0 if out of bounds
                    body.append(("alu", ("*", tmp_idx[u], tmp_idx[u], idx_tmp1[u])))
                for u in range(actual_unroll):
                    i = base_i + u
                    body.append(("debug", ("compare", tmp_idx[u], (round, i, "wrapped_idx"))))

                # Phase 6: Store results - parallel collapse
                for u in range(actual_unroll):
                    i = base_i + u
                    i_const = i_constants[i]
                    body.append(("alu", ("+", tmp_addr[u], self.scratch["inp_indices_p"], i_const)))
                for u in range(actual_unroll):
                    body.append(("store", ("store", tmp_addr[u], tmp_idx[u])))
                for u in range(actual_unroll):
                    i = base_i + u
                    i_const = i_constants[i]
                    body.append(("alu", ("+", tmp_addr[u], self.scratch["inp_values_p"], i_const)))
                for u in range(actual_unroll):
                    body.append(("store", ("store", tmp_addr[u], tmp_val[u])))

        body_instrs = self.build(body, vliw=True)  # Enable dependency-aware packing
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Use the scalar packed version - simpler and works.
        """
        return self.build_kernel_scalar_packed(forest_height, n_nodes, batch_size, rounds)

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
