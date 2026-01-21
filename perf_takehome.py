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
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                _, dest, src = slot
                writes.update(range(dest, dest + 8))
                reads.add(src)
            elif slot[0] == "multiply_add":
                _, dest, a, b, c = slot
                writes.update(range(dest, dest + 8))
                reads.update(range(a, a + 8))
                reads.update(range(b, b + 8))
                reads.update(range(c, c + 8))
            else:
                # Standard 3-operand vector ALU
                _, dest, src1, src2 = slot
                writes.update(range(dest, dest + 8))
                reads.update(range(src1, src1 + 8))
                reads.update(range(src2, src2 + 8))
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
            # Debug operations should respect read dependencies
            if slot[0] == "compare":
                _, loc, _ = slot
                reads.add(loc)
            elif slot[0] == "vcompare":
                _, loc, _ = slot
                reads.update(range(loc, loc + 8))
        
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
        Vectorized kernel with aggressive unrolling for maximum VLIW packing.
        Process multiple VLEN=8 vectors at once to maximize ILP.
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
        self.add("debug", ("comment", "Starting vectorized loop"))

        body = []  # array of slots

        # Unroll factor for vector iterations (process this many VLEN-vectors at once)
        VECTOR_UNROLL = 16  # Process 16*8=128 elements at once for maximum ILP
        
        # Allocate vector register sets for unrolled iterations
        unroll_vregs = []
        for u in range(VECTOR_UNROLL):
            unroll_vregs.append({
                'idx': self.alloc_scratch(f"vu{u}_idx", VLEN),
                'val': self.alloc_scratch(f"vu{u}_val", VLEN),
                'node_val': self.alloc_scratch(f"vu{u}_node_val", VLEN),
                'tmp1': self.alloc_scratch(f"vu{u}_tmp1", VLEN),
                'tmp2': self.alloc_scratch(f"vu{u}_tmp2", VLEN),
                'tmp3': self.alloc_scratch(f"vu{u}_tmp3", VLEN),
                'addr': self.alloc_scratch(f"vu{u}_addr"),
            })
        
        # Scalar temporaries for addressing
        addr_tmp = self.alloc_scratch("addr_tmp")
        
        # Broadcast constants to vectors (allocate once, reuse)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        
        # Pre-allocate hash constant vectors (reuse across iterations)
        hash_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_vecs.append({
                'val1': self.alloc_scratch(f"hash{hi}_val1", VLEN),
                'val3': self.alloc_scratch(f"hash{hi}_val3", VLEN),
            })
        
        # Broadcast constants once
        body.append(("valu", ("vbroadcast", v_zero, zero_const)))
        body.append(("valu", ("vbroadcast", v_two, two_const)))
        body.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))
        
        # Broadcast hash constants once
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            body.append(("valu", ("vbroadcast", hash_vecs[hi]['val1'], self.scratch_const(val1))))
            body.append(("valu", ("vbroadcast", hash_vecs[hi]['val3'], self.scratch_const(val3))))
        
        for round in range(rounds):
            # Process in vector chunks, unrolling by VECTOR_UNROLL
            for i_base in range(0, batch_size, VLEN * VECTOR_UNROLL):
                effective_unroll = min(VECTOR_UNROLL, (batch_size - i_base + VLEN - 1) // VLEN)
                
                if effective_unroll == VECTOR_UNROLL:
                    # Full unrolled vector processing
                    
                    # Stage 1: Load all indices (all unrolled iterations)
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        i_const = self.scratch_const(i_offset)
                        reg = unroll_vregs[u]
                        body.append(("alu", ("+", reg['addr'], self.scratch["inp_indices_p"], i_const)))
                    for u in range(effective_unroll):
                        reg = unroll_vregs[u]
                        body.append(("load", ("vload", reg['idx'], reg['addr'])))
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        reg = unroll_vregs[u]
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", reg['idx'] + vi, (round, i_offset + vi, "idx"))))
                    
                    # Stage 2: Load all values
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        i_const = self.scratch_const(i_offset)
                        reg = unroll_vregs[u]
                        body.append(("alu", ("+", reg['addr'], self.scratch["inp_values_p"], i_const)))
                    for u in range(effective_unroll):
                        reg = unroll_vregs[u]
                        body.append(("load", ("vload", reg['val'], reg['addr'])))
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        reg = unroll_vregs[u]
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", reg['val'] + vi, (round, i_offset + vi, "val"))))
                    
                    # Stage 3: Irregular loads for forest_values (interleave across unrolled iterations)
                    for vi in range(VLEN):
                        for u in range(effective_unroll):
                            i_offset = i_base + u * VLEN
                            reg = unroll_vregs[u]
                            body.append(("alu", ("+", addr_tmp, self.scratch["forest_values_p"], reg['idx'] + vi)))
                            body.append(("load", ("load", reg['node_val'] + vi, addr_tmp)))
                            body.append(("debug", ("compare", reg['node_val'] + vi, (round, i_offset + vi, "node_val"))))
                    
                    # Stage 4: Vector XOR (all unrolled iterations)
                    for u in range(effective_unroll):
                        reg = unroll_vregs[u]
                        body.append(("valu", ("^", reg['val'], reg['val'], reg['node_val'])))
                    
                    # Stage 5: Vector hash operations (interleave across unrolled iterations)
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        for u in range(effective_unroll):
                            reg = unroll_vregs[u]
                            body.append(("valu", (op1, reg['tmp1'], reg['val'], hash_vecs[hi]['val1'])))
                        for u in range(effective_unroll):
                            reg = unroll_vregs[u]
                            body.append(("valu", (op3, reg['tmp2'], reg['val'], hash_vecs[hi]['val3'])))
                        for u in range(effective_unroll):
                            i_offset = i_base + u * VLEN
                            reg = unroll_vregs[u]
                            body.append(("valu", (op2, reg['val'], reg['tmp1'], reg['tmp2'])))
                            for vi in range(VLEN):
                                body.append(("debug", ("compare", reg['val'] + vi, (round, i_offset + vi, "hash_stage", hi))))
                    
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        reg = unroll_vregs[u]
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", reg['val'] + vi, (round, i_offset + vi, "hashed_val"))))
                    
                    # Stage 6-9: Index update and bounds checking
                    for u in range(effective_unroll):
                        reg = unroll_vregs[u]
                        body.append(("valu", ("%", reg['tmp1'], reg['val'], v_two)))
                        body.append(("valu", ("==", reg['tmp1'], reg['tmp1'], v_zero)))
                        body.append(("valu", ("-", reg['tmp3'], v_two, reg['tmp1'])))
                        body.append(("valu", ("*", reg['idx'], reg['idx'], v_two)))
                        body.append(("valu", ("+", reg['idx'], reg['idx'], reg['tmp3'])))
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        reg = unroll_vregs[u]
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", reg['idx'] + vi, (round, i_offset + vi, "next_idx"))))
                    
                    for u in range(effective_unroll):
                        reg = unroll_vregs[u]
                        body.append(("valu", ("<", reg['tmp1'], reg['idx'], v_n_nodes)))
                        body.append(("valu", ("*", reg['idx'], reg['tmp1'], reg['idx'])))
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        reg = unroll_vregs[u]
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", reg['idx'] + vi, (round, i_offset + vi, "wrapped_idx"))))
                    
                    # Stage 10: Store results (all unrolled iterations)
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        i_const = self.scratch_const(i_offset)
                        reg = unroll_vregs[u]
                        body.append(("alu", ("+", reg['addr'], self.scratch["inp_indices_p"], i_const)))
                        body.append(("store", ("vstore", reg['addr'], reg['idx'])))
                        body.append(("alu", ("+", reg['addr'], self.scratch["inp_values_p"], i_const)))
                        body.append(("store", ("vstore", reg['addr'], reg['val'])))
                else:
                    # Partial unroll or scalar tail loop
                    for u in range(effective_unroll):
                        i_offset = i_base + u * VLEN
                        if i_offset + VLEN <= batch_size:
                            # Full vector
                            i_const = self.scratch_const(i_offset)
                            reg = unroll_vregs[0]  # Reuse first register set
                            
                            body.append(("alu", ("+", reg['addr'], self.scratch["inp_indices_p"], i_const)))
                            body.append(("load", ("vload", reg['idx'], reg['addr'])))
                            for vi in range(VLEN):
                                body.append(("debug", ("compare", reg['idx'] + vi, (round, i_offset + vi, "idx"))))
                            
                            body.append(("alu", ("+", reg['addr'], self.scratch["inp_values_p"], i_const)))
                            body.append(("load", ("vload", reg['val'], reg['addr'])))
                            for vi in range(VLEN):
                                body.append(("debug", ("compare", reg['val'] + vi, (round, i_offset + vi, "val"))))
                            
                            for vi in range(VLEN):
                                body.append(("alu", ("+", addr_tmp, self.scratch["forest_values_p"], reg['idx'] + vi)))
                                body.append(("load", ("load", reg['node_val'] + vi, addr_tmp)))
                                body.append(("debug", ("compare", reg['node_val'] + vi, (round, i_offset + vi, "node_val"))))
                            
                            body.append(("valu", ("^", reg['val'], reg['val'], reg['node_val'])))
                            
                            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                                body.append(("valu", (op1, reg['tmp1'], reg['val'], hash_vecs[hi]['val1'])))
                                body.append(("valu", (op3, reg['tmp2'], reg['val'], hash_vecs[hi]['val3'])))
                                body.append(("valu", (op2, reg['val'], reg['tmp1'], reg['tmp2'])))
                                for vi in range(VLEN):
                                    body.append(("debug", ("compare", reg['val'] + vi, (round, i_offset + vi, "hash_stage", hi))))
                            
                            for vi in range(VLEN):
                                body.append(("debug", ("compare", reg['val'] + vi, (round, i_offset + vi, "hashed_val"))))
                            
                            body.append(("valu", ("%", reg['tmp1'], reg['val'], v_two)))
                            body.append(("valu", ("==", reg['tmp1'], reg['tmp1'], v_zero)))
                            body.append(("valu", ("-", reg['tmp3'], v_two, reg['tmp1'])))
                            body.append(("valu", ("*", reg['idx'], reg['idx'], v_two)))
                            body.append(("valu", ("+", reg['idx'], reg['idx'], reg['tmp3'])))
                            for vi in range(VLEN):
                                body.append(("debug", ("compare", reg['idx'] + vi, (round, i_offset + vi, "next_idx"))))
                            
                            body.append(("valu", ("<", reg['tmp1'], reg['idx'], v_n_nodes)))
                            body.append(("valu", ("*", reg['idx'], reg['tmp1'], reg['idx'])))
                            for vi in range(VLEN):
                                body.append(("debug", ("compare", reg['idx'] + vi, (round, i_offset + vi, "wrapped_idx"))))
                            
                            body.append(("alu", ("+", reg['addr'], self.scratch["inp_indices_p"], i_const)))
                            body.append(("store", ("vstore", reg['addr'], reg['idx'])))
                            body.append(("alu", ("+", reg['addr'], self.scratch["inp_values_p"], i_const)))
                            body.append(("store", ("vstore", reg['addr'], reg['val'])))
                        else:
                            # Scalar tail
                            for i in range(i_offset, min(i_offset + VLEN, batch_size)):
                                i_const = self.scratch_const(i)
                                
                                body.append(("alu", ("+", addr_tmp, self.scratch["inp_indices_p"], i_const)))
                                body.append(("load", ("load", tmp1, addr_tmp)))
                                body.append(("debug", ("compare", tmp1, (round, i, "idx"))))
                                
                                body.append(("alu", ("+", addr_tmp, self.scratch["inp_values_p"], i_const)))
                                body.append(("load", ("load", tmp2, addr_tmp)))
                                body.append(("debug", ("compare", tmp2, (round, i, "val"))))
                                
                                body.append(("alu", ("+", addr_tmp, self.scratch["forest_values_p"], tmp1)))
                                body.append(("load", ("load", tmp3, addr_tmp)))
                                body.append(("debug", ("compare", tmp3, (round, i, "node_val"))))
                                
                                body.append(("alu", ("^", tmp2, tmp2, tmp3)))
                                
                                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                                    val1_const = self.scratch_const(val1)
                                    val3_const = self.scratch_const(val3)
                                    body.append(("alu", (op1, tmp3, tmp2, val1_const)))
                                    body.append(("alu", (op3, addr_tmp, tmp2, val3_const)))
                                    body.append(("alu", (op2, tmp2, tmp3, addr_tmp)))
                                    body.append(("debug", ("compare", tmp2, (round, i, "hash_stage", hi))))
                                
                                body.append(("debug", ("compare", tmp2, (round, i, "hashed_val"))))
                                
                                body.append(("alu", ("%", tmp3, tmp2, two_const)))
                                body.append(("alu", ("==", tmp3, tmp3, zero_const)))
                                body.append(("alu", ("-", tmp3, two_const, tmp3)))
                                body.append(("alu", ("*", tmp1, tmp1, two_const)))
                                body.append(("alu", ("+", tmp1, tmp1, tmp3)))
                                body.append(("debug", ("compare", tmp1, (round, i, "next_idx"))))
                                
                                body.append(("alu", ("<", tmp3, tmp1, self.scratch["n_nodes"])))
                                body.append(("alu", ("*", tmp1, tmp3, tmp1)))
                                body.append(("debug", ("compare", tmp1, (round, i, "wrapped_idx"))))
                                
                                body.append(("alu", ("+", addr_tmp, self.scratch["inp_indices_p"], i_const)))
                                body.append(("store", ("store", addr_tmp, tmp1)))
                                body.append(("alu", ("+", addr_tmp, self.scratch["inp_values_p"], i_const)))
                                body.append(("store", ("store", addr_tmp, tmp2)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
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
