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


def _vec_range(base: int, length: int = VLEN) -> range:
    """Helper to get range of vector element addresses."""
    return range(base, base + length)


def _slot_rw(engine: str, slot: tuple) -> tuple[list[int], list[int]]:
    """Get read and write addresses for a slot (for dependency analysis)."""
    reads: list[int] = []
    writes: list[int] = []

    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]
    elif engine == "valu":
        match slot:
            case ("vbroadcast", dest, src):
                reads = [src]
                writes = list(_vec_range(dest))
            case ("multiply_add", dest, a, b, c):
                reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
                writes = list(_vec_range(dest))
            case (_op, dest, a1, a2):
                reads = list(_vec_range(a1)) + list(_vec_range(a2))
                writes = list(_vec_range(dest))
            case _:
                raise NotImplementedError(f"Unknown valu op {slot}")
    elif engine == "load":
        match slot:
            case ("load", dest, addr):
                reads = [addr]
                writes = [dest]
            case ("vload", dest, addr):
                reads = [addr]
                writes = list(_vec_range(dest))
            case ("const", dest, _val):
                writes = [dest]
            case _:
                raise NotImplementedError(f"Unknown load op {slot}")
    elif engine == "store":
        match slot:
            case ("store", addr, src):
                reads = [addr, src]
            case ("vstore", addr, src):
                reads = [addr] + list(_vec_range(src))
            case _:
                raise NotImplementedError(f"Unknown store op {slot}")
    elif engine == "flow":
        match slot:
            case ("select", dest, cond, a, b):
                reads = [cond, a, b]
                writes = [dest]
            case ("vselect", dest, cond, a, b):
                reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
                writes = list(_vec_range(dest))
            case ("halt",) | ("pause",):
                pass
            case _:
                raise NotImplementedError(f"Unknown flow op {slot}")

    return reads, writes


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """Automatically schedule operations into VLIW bundles respecting dependencies."""
    cycles: list[dict[str, list[tuple]]] = []
    usage: list[dict[str, int]] = []
    ready_time: dict[int, int] = defaultdict(int)
    last_write: dict[int, int] = defaultdict(lambda: -1)
    last_read: dict[int, int] = defaultdict(lambda: -1)

    def ensure_cycle(cycle: int) -> None:
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine: str, earliest: int) -> int:
        cycle = earliest
        limit = SLOT_LIMITS[engine]
        while True:
            ensure_cycle(cycle)
            if usage[cycle][engine] < limit:
                return cycle
            cycle += 1

    for engine, slot in slots:
        if engine == "debug":
            continue  # Skip debug instructions
        reads, writes = _slot_rw(engine, slot)
        earliest = 0
        for addr in reads:
            earliest = max(earliest, ready_time[addr])
        for addr in writes:
            earliest = max(earliest, last_write[addr] + 1, last_read[addr])

        cycle = find_cycle(engine, earliest)
        ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot)
        usage[cycle][engine] += 1

        for addr in reads:
            if last_read[addr] < cycle:
                last_read[addr] = cycle
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1

    return [c for c in cycles if c]


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

    def scratch_const(self, val, name=None, slots=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            if slots is None:
                self.add("load", ("const", addr, val))
            else:
                slots.append(("load", ("const", addr, val)))
            self.const_map[val] = addr
        return self.const_map[val]

    def alloc_vec(self, name=None):
        """Allocate a vector (VLEN elements) in scratch."""
        return self.alloc_scratch(name, VLEN)

    def scratch_vconst(self, val, name=None, slots=None):
        """Allocate and broadcast a constant to a vector."""
        if val not in self.vconst_map:
            scalar = self.scratch_const(val, slots=slots)
            addr = self.alloc_vec(name)
            if slots is None:
                self.add("valu", ("vbroadcast", addr, scalar))
            else:
                slots.append(("valu", ("vbroadcast", addr, scalar)))
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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel with automatic VLIW scheduling.
        Uses VLEN=8 to process 8 elements per vector operation.
        """
        # Scalar temporaries
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_scalar = self.alloc_scratch("tmp_scalar")

        # Initialize from memory header
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        init_slots = []
        for i, v in enumerate(init_vars):
            init_slots.append(("load", ("const", tmp_scalar, i)))
            init_slots.append(("load", ("load", self.scratch[v], tmp_scalar)))

        # Vector constants
        zero_vec = self.scratch_vconst(0, "v_zero", init_slots)
        one_vec = self.scratch_vconst(1, "v_one", init_slots)
        two_vec = self.scratch_vconst(2, "v_two", init_slots)
        one_const = self.scratch_const(1, slots=init_slots)

        # Broadcast forest_values_p to vector for address calculation
        forest_vec = self.alloc_vec("v_forest_p")
        init_slots.append(("valu", ("vbroadcast", forest_vec, self.scratch["forest_values_p"])))

        # Broadcast n_nodes to vector for comparison
        n_nodes_vec = self.alloc_vec("v_n_nodes")
        init_slots.append(("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])))

        # Hash constants (vectorized)
        hash_vec_consts1 = []
        hash_vec_consts3 = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_vec_consts1.append(self.scratch_vconst(val1, slots=init_slots))
            hash_vec_consts3.append(self.scratch_vconst(val3, slots=init_slots))

        # Schedule init phase
        self.instrs.extend(_schedule_slots(init_slots))
        self.add("flow", ("pause",))

        # Main body
        assert batch_size % VLEN == 0
        n_blocks = batch_size // VLEN

        # Allocate scratch for all idx/val vectors (persistent across rounds)
        idx_base = self.alloc_scratch("idx_scratch", batch_size)
        val_base = self.alloc_scratch("val_scratch", batch_size)

        # Allocate temporary vectors
        tmp1_vec = self.alloc_vec("tmp1_vec")
        tmp2_vec = self.alloc_vec("tmp2_vec")
        node_vec = self.alloc_vec("node_vec")

        slots: list[tuple[str, tuple]] = []

        # Load initial idx/val from memory
        offset_const = self.scratch_const(0, slots=slots)
        vlen_const = self.scratch_const(VLEN, slots=slots)
        offset = self.alloc_scratch("offset")
        slots.append(("load", ("const", offset, 0)))

        for block in range(n_blocks):
            # Load idx vector: vload from inp_indices_p + block*VLEN
            slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], offset)))
            slots.append(("load", ("vload", idx_base + block * VLEN, tmp_addr)))
            # Load val vector: vload from inp_values_p + block*VLEN
            slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], offset)))
            slots.append(("load", ("vload", val_base + block * VLEN, tmp_addr)))
            # Increment offset
            slots.append(("alu", ("+", offset, offset, vlen_const)))

        # Process all rounds for all blocks
        for block in range(n_blocks):
            idx_vec = idx_base + block * VLEN
            val_vec = val_base + block * VLEN

            for rnd in range(rounds):
                # Gather node values: node_val = mem[forest_values_p + idx]
                # Use scalar loads for gather (no native vector gather)
                for lane in range(VLEN):
                    slots.append(("alu", ("+", tmp1_vec + lane, forest_vec + lane, idx_vec + lane)))
                for lane in range(VLEN):
                    slots.append(("load", ("load", node_vec + lane, tmp1_vec + lane)))

                # val = val ^ node_val
                slots.append(("valu", ("^", val_vec, val_vec, node_vec)))

                # Hash computation (6 stages)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    slots.append(("valu", (op1, tmp1_vec, val_vec, hash_vec_consts1[hi])))
                    slots.append(("valu", (op3, tmp2_vec, val_vec, hash_vec_consts3[hi])))
                    slots.append(("valu", (op2, val_vec, tmp1_vec, tmp2_vec)))

                # idx = 2*idx + (1 if val%2==0 else 2)
                # tmp1 = val & 1 (same as val % 2)
                slots.append(("valu", ("&", tmp1_vec, val_vec, one_vec)))
                # tmp1 = (tmp1 == 0) ? 1 : 0
                slots.append(("valu", ("==", tmp1_vec, tmp1_vec, zero_vec)))
                # tmp2 = tmp1 ? 1 : 2 (vselect)
                slots.append(("flow", ("vselect", tmp2_vec, tmp1_vec, one_vec, two_vec)))
                # idx = idx * 2
                slots.append(("valu", ("*", idx_vec, idx_vec, two_vec)))
                # idx = idx + tmp2
                slots.append(("valu", ("+", idx_vec, idx_vec, tmp2_vec)))

                # idx = 0 if idx >= n_nodes else idx
                slots.append(("valu", ("<", tmp1_vec, idx_vec, n_nodes_vec)))
                slots.append(("flow", ("vselect", idx_vec, tmp1_vec, idx_vec, zero_vec)))

        # Store final results
        slots.append(("load", ("const", offset, 0)))
        for block in range(n_blocks):
            # Store val vector
            slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], offset)))
            slots.append(("store", ("vstore", tmp_addr, val_base + block * VLEN)))
            # Store idx vector (optional but keeping for correctness)
            slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], offset)))
            slots.append(("store", ("vstore", tmp_addr, idx_base + block * VLEN)))
            # Increment offset
            slots.append(("alu", ("+", offset, offset, vlen_const)))

        # Schedule all body operations
        self.instrs.extend(_schedule_slots(slots))
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
