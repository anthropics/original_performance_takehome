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

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """
        Quasicrystal aperiodic VLIW packing with dependency tracking.
        Pack multiple independent instructions into single cycles while
        respecting SLOT_LIMITS and register dependencies.
        """
        if not vliw:
            return [{engine: [slot]} for engine, slot in slots]
        
        instrs = []
        current_bundle = {}
        slot_counts = {engine: 0 for engine in SLOT_LIMITS}
        written_this_cycle = set()
        
        for engine, slot in slots:
            reads, writes = self._analyze_dependencies(engine, slot)
            
            # Can add if: slot available AND no read-after-write hazard
            can_add = (
                slot_counts[engine] < SLOT_LIMITS[engine] and
                not (reads & written_this_cycle)
            )
            
            if not can_add:
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {}
                slot_counts = {engine: 0 for engine in SLOT_LIMITS}
                written_this_cycle = set()
            
            if engine not in current_bundle:
                current_bundle[engine] = []
            current_bundle[engine].append(slot)
            slot_counts[engine] += 1
            written_this_cycle.update(writes)
        
        if current_bundle:
            instrs.append(current_bundle)
        
        return instrs

    def _analyze_dependencies(self, engine: str, slot: tuple):
        """Extract read/write register addresses from instruction slots"""
        reads = set()
        writes = set()
        
        if engine == "alu":
            op, dest, src1, src2 = slot
            writes.add(dest)
            reads.update([src1, src2])
        elif engine == "load":
            if slot[0] == "load":
                _, dest, addr = slot
                writes.add(dest)
                reads.add(addr)
            elif slot[0] == "const":
                _, dest, _ = slot
                writes.add(dest)
            elif slot[0] == "vload":
                _, dest, addr = slot
                writes.update(range(dest, dest + 8))  # VLEN=8
                reads.add(addr)
        elif engine == "store":
            if slot[0] == "store":
                _, addr, val = slot
                reads.update([addr, val])
            elif slot[0] == "vstore":
                _, addr, src = slot
                reads.add(addr)
                reads.update(range(src, src + 8))
        elif engine == "flow":
            if slot[0] == "select":
                _, dest, cond, a, b = slot
                writes.add(dest)
                reads.update([cond, a, b])
            elif slot[0] == "vselect":
                _, dest, cond, a, b = slot
                writes.update(range(dest, dest + 8))
                reads.update(range(cond, cond + 8))
                reads.update(range(a, a + 8))
                reads.update(range(b, b + 8))
        elif engine == "debug":
            # Debug operations are disabled in submission tests, so we can
            # pack them freely without considering dependencies
            pass
        
        return reads, writes

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
        """
        Phason-flipped hash: group operations by type for VLIW explosion.
        ALU engine has 12 slots - maximize parallel execution.
        Reuse tmp1/tmp2 to minimize register pressure.
        """
        slots = []
        
        # Phase 1: All op1 operations (can execute in parallel - up to 12)
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

        # Fibonacci unrolling with separate temporaries per iteration
        UNROLL_FACTOR = 16  # Optimal unroll factor for this architecture
        
        # Allocate independent register sets for each unrolled iteration
        unroll_regs = []
        for u in range(UNROLL_FACTOR):
            unroll_regs.append({
                'idx': self.alloc_scratch(f"u{u}_idx"),
                'val': self.alloc_scratch(f"u{u}_val"),
                'node_val': self.alloc_scratch(f"u{u}_node_val"),
                'addr': self.alloc_scratch(f"u{u}_addr"),
                'addr2': self.alloc_scratch(f"u{u}_addr2"),  # Additional addr for stores
                'tmp1': self.alloc_scratch(f"u{u}_tmp1"),
                'tmp2': self.alloc_scratch(f"u{u}_tmp2"),
                'tmp3': self.alloc_scratch(f"u{u}_tmp3"),
            })
        
        for round in range(rounds):
            for i_base in range(0, batch_size, UNROLL_FACTOR):
                effective_unroll = min(UNROLL_FACTOR, batch_size - i_base)
                
                # Stage 1: Load all indices in parallel
                for u in range(effective_unroll):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    reg = unroll_regs[u]
                    body.append(("alu", ("+", reg['addr'], self.scratch["inp_indices_p"], i_const)))
                for u in range(effective_unroll):
                    reg = unroll_regs[u]
                    body.append(("load", ("load", reg['idx'], reg['addr'])))
                    body.append(("debug", ("compare", reg['idx'], (round, i_base + u, "idx"))))
                
                # Stage 2: Load all values in parallel
                for u in range(effective_unroll):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    reg = unroll_regs[u]
                    body.append(("alu", ("+", reg['addr'], self.scratch["inp_values_p"], i_const)))
                for u in range(effective_unroll):
                    reg = unroll_regs[u]
                    body.append(("load", ("load", reg['val'], reg['addr'])))
                    body.append(("debug", ("compare", reg['val'], (round, i_base + u, "val"))))
                
                # Stage 3: Load all node values in parallel
                for u in range(effective_unroll):
                    reg = unroll_regs[u]
                    body.append(("alu", ("+", reg['addr'], self.scratch["forest_values_p"], reg['idx'])))
                for u in range(effective_unroll):
                    reg = unroll_regs[u]
                    body.append(("load", ("load", reg['node_val'], reg['addr'])))
                    body.append(("debug", ("compare", reg['node_val'], (round, i_base + u, "node_val"))))
                
                # Stage 4: XOR operations (can all happen in parallel)
                for u in range(effective_unroll):
                    i = i_base + u
                    reg = unroll_regs[u]
                    body.append(("alu", ("^", reg['val'], reg['val'], reg['node_val'])))
                
                # Stage 5: Hash operations (interleaved across iterations for ILP)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    val1_const = self.scratch_const(val1)
                    val3_const = self.scratch_const(val3)
                    # All op1 operations for this stage across all iterations
                    for u in range(effective_unroll):
                        i = i_base + u
                        reg = unroll_regs[u]
                        body.append(("alu", (op1, reg['tmp1'], reg['val'], val1_const)))
                    # All op3 operations for this stage across all iterations
                    for u in range(effective_unroll):
                        i = i_base + u
                        reg = unroll_regs[u]
                        body.append(("alu", (op3, reg['tmp2'], reg['val'], val3_const)))
                    # All op2 operations for this stage across all iterations
                    for u in range(effective_unroll):
                        i = i_base + u
                        reg = unroll_regs[u]
                        body.append(("alu", (op2, reg['val'], reg['tmp1'], reg['tmp2'])))
                        body.append(("debug", ("compare", reg['val'], (round, i, "hash_stage", hi))))
                # Final hash_val debug compares
                for u in range(effective_unroll):
                    i = i_base + u
                    reg = unroll_regs[u]
                    body.append(("debug", ("compare", reg['val'], (round, i, "hashed_val"))))
                
                # Stage 6: Index update arithmetic (parallel where possible)
                for u in range(effective_unroll):
                    i = i_base + u
                    reg = unroll_regs[u]
                    body.append(("alu", ("%", reg['tmp1'], reg['val'], two_const)))
                    body.append(("alu", ("==", reg['tmp1'], reg['tmp1'], zero_const)))
                
                # Stage 7: Conditional select (arithmetic equivalent - Phase 9-12)
                for u in range(effective_unroll):
                    i = i_base + u
                    reg = unroll_regs[u]
                    # Replace flow select with arithmetic: tmp3 = 2 - tmp1
                    body.append(("alu", ("-", reg['tmp3'], two_const, reg['tmp1'])))
                
                # Stage 8: Index calculations
                for u in range(effective_unroll):
                    i = i_base + u
                    reg = unroll_regs[u]
                    body.append(("alu", ("*", reg['idx'], reg['idx'], two_const)))
                    body.append(("alu", ("+", reg['idx'], reg['idx'], reg['tmp3'])))
                    body.append(("debug", ("compare", reg['idx'], (round, i, "next_idx"))))
                
                # Stage 9: Bounds checking
                for u in range(effective_unroll):
                    i = i_base + u
                    reg = unroll_regs[u]
                    body.append(("alu", ("<", reg['tmp1'], reg['idx'], self.scratch["n_nodes"])))
                    # Replace flow select with arithmetic: idx = tmp1 * idx
                    body.append(("alu", ("*", reg['idx'], reg['tmp1'], reg['idx'])))
                    body.append(("debug", ("compare", reg['idx'], (round, i, "wrapped_idx"))))
                
                # Stage 10: Store results - compute all addresses first, then store
                # Address calculations for idx stores
                for u in range(effective_unroll):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    reg = unroll_regs[u]
                    body.append(("alu", ("+", reg['addr'], self.scratch["inp_indices_p"], i_const)))
                # Address calculations for val stores
                for u in range(effective_unroll):
                    i = i_base + u
                    i_const = self.scratch_const(i)
                    reg = unroll_regs[u]
                    body.append(("alu", ("+", reg['addr2'], self.scratch["inp_values_p"], i_const)))
                # All stores together (interleave idx and val to hit 2 stores per cycle)
                for u in range(effective_unroll):
                    reg = unroll_regs[u]
                    body.append(("store", ("store", reg['addr'], reg['idx'])))
                    body.append(("store", ("store", reg['addr2'], reg['val'])))

        body_instrs = self.build(body, vliw=True)
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
