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
from dataclasses import dataclass

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

@dataclass
class Op:
    engine: Engine
    args: tuple
    # For dependency tracking (simple counter based)
    latency: int = 1

class SmartScheduler:
    def __init__(self, limits):
        self.limits = limits
        self.bundles = [] # List of dicts
        
    def schedule(self, op: Op, min_time: int) -> int:
        """
        Schedule op at earliest time t >= min_time satisfying slot limits.
        Returns t.
        """
        t = min_time
        while True:
            # Expand bundles if needed
            while len(self.bundles) <= t:
                self.bundles.append(defaultdict(list))
            
            bundle = self.bundles[t]
            current_count = len(bundle[op.engine])
            limit = self.limits.get(op.engine, 0)
            
            if current_count < limit:
                bundle[op.engine].append(op.args)
                return t
            t += 1

    def get_instrs(self):
        # Convert defaultdicts to dicts for the simulator
        return [dict(b) for b in self.bundles]

class OptimizedKernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        
        # Scratch Layout
        self.IDX_BASE = 0
        self.VAL_BASE = 256
        self.CACHE_BASE = 512
        self.CACHE_LIMIT = 511
        self.TMP_BASE = 1024
        self.scratch_ptr = 1024

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.instrs.insert(0, {"load": [("const", addr, val)]})
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        # Constants
        ZERO = self.scratch_const(0, "zero")
        ONE = self.scratch_const(1, "one")
        TWO = self.scratch_const(2, "two")
        
        def make_vec_const(val, name):
            s = self.scratch_const(val)
            v = self.alloc_scratch(name, VLEN)
            self.instrs.append({"valu": [("vbroadcast", v, s)]})
            return v
            
        ZERO_VEC = make_vec_const(0, "zero_vec")
        ONE_VEC = make_vec_const(1, "one_vec")
        TWO_VEC = make_vec_const(2, "two_vec")

        # Prologue
        inp_indices_p_addr = self.alloc_scratch("inp_indices_p")
        inp_values_p_addr = self.alloc_scratch("inp_values_p")
        forest_values_p_addr = self.alloc_scratch("forest_values_p")
        n_nodes_addr = self.alloc_scratch("n_nodes")
        tmp = self.alloc_scratch("tmp") 
        
        prologue_instrs = []
        prologue_instrs.append({"load": [("const", tmp, 5)]})
        prologue_instrs.append({"load": [("load", inp_indices_p_addr, tmp)]})
        prologue_instrs.append({"load": [("const", tmp, 6)]})
        prologue_instrs.append({"load": [("load", inp_values_p_addr, tmp)]})
        prologue_instrs.append({"load": [("const", tmp, 4)]})
        prologue_instrs.append({"load": [("load", forest_values_p_addr, tmp)]})
        prologue_instrs.append({"load": [("const", tmp, 1)]})
        prologue_instrs.append({"load": [("load", n_nodes_addr, tmp)]})
        
        self.instrs.extend(prologue_instrs)

        # Allocate shared constants in CACHE_BASE to save TMP_BASE for batch temps
        self.scratch_ptr = self.CACHE_BASE
        
        # Broadcast n_nodes immediately
        N_NODES_VEC = self.alloc_scratch("n_nodes_vec", VLEN)
        self.instrs.append({"valu": [("vbroadcast", N_NODES_VEC, n_nodes_addr)]})
        
        # Load Indices and Values loops
        load_loop = []
        for b in range(0, batch_size, VLEN):
            b_const = self.scratch_const(b)
            addr_reg = self.alloc_scratch("addr")
            load_loop.append({"alu": [("+", addr_reg, inp_indices_p_addr, b_const)]})
            load_loop.append({"load": [("vload", self.IDX_BASE + b, addr_reg)]})
            load_loop.append({"alu": [("+", addr_reg, inp_values_p_addr, b_const)]})
            load_loop.append({"load": [("vload", self.VAL_BASE + b, addr_reg)]})
        self.instrs.extend(load_loop)
        
        # Hash Constants broadcasting
        HASH_CONSTS_VECS = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
             c1 = self.alloc_scratch(f"h{hi}_c1", VLEN)
             c3 = self.alloc_scratch(f"h{hi}_c3", VLEN)
             s1 = self.scratch_const(val1)
             s3 = self.scratch_const(val3)
             self.instrs.append({"valu": [("vbroadcast", c1, s1), ("vbroadcast", c3, s3)]})
             HASH_CONSTS_VECS.append((c1, c3))
             
        # Reset scratch_ptr to TMP_BASE for batch temporary allocations
        # This keeps hash constants in CACHE_BASE (512-~620) and batch temps in TMP_BASE (1024-1535)
        self.scratch_ptr = self.TMP_BASE

        # Smart Scheduler for Main Loop
        scheduler = SmartScheduler(SLOT_LIMITS)
        batch_times = defaultdict(int) # tid -> time available
        
        # Waves: Limit concurrently active separate threads
        # But we want pipelining.
        # Can we run ALL 32 batches pipelined?
        # Each batch needs 32 words temp.
        # 32 * 32 = 1024 words.
        # We have 512 words.
        # So Max 16 threads.
        # So we MUST do waves.
        # Wave 1: Batch 0-15.
        # Wave 2: Batch 16-31.
        
        THREAD_MEM_START = self.scratch_ptr
        
        for wave_start in [0, 16 * VLEN]:
            if wave_start >= batch_size: break
            
            # Map batch -> thread_id (0..15)
            # We must run these threads through ALL rounds?
            # NO. Rounds must be sequential for a batch.
            # But Wave 1 can finish Round 0..15.
            # Then Wave 2 data is loaded?
            # Wait. "Load Indices/Values" at start loaded ALL 32 batches.
            # So data is in scratch.
            # So we can fully finish Batches 0..15 (All Rounds).
            # THEN run Batches 16..31 (All Rounds).
            # This is simpler and cache-friendly!
            # It also minimizes "batch_times" tracking.
            
            wave_generators = []
            for b_idx in range(16):
                b_off = wave_start + b_idx * VLEN
                if b_off >= batch_size: break
                
                # Create generator for this batch for ALL ROUNDS
                gen = self.gen_batch_all_rounds(
                    b_off, rounds, HASH_CONSTS_VECS, 
                    forest_values_p_addr, 

                    TWO_VEC, ZERO_VEC, ONE_VEC, N_NODES_VEC,
                    THREAD_MEM_START + (b_idx * 32)
                )
                wave_generators.append((b_idx, gen))
                
            # Run scheduler for this wave
            has_work = True
            while has_work:
                has_work = False
                for tid, gen in wave_generators:
                    try:
                        op_list = next(gen)
                        has_work = True
                        
                        start_time = batch_times[tid]
                        max_t = start_time
                        
                        for op in op_list:
                            t = scheduler.schedule(op, start_time)
                            max_t = max(max_t, t + op.latency)
                        
                        batch_times[tid] = max_t
                        
                    except StopIteration:
                        pass
            
            # Reset batch_times for next wave? 
            # Yes, new wave can start whenever.
            # Ideally immediately.
            # But effectively we just continue scheduling.
            # We can map tid 0 of wave 2 to tid 0 of wave 1 registers.
            # So we reuse temps.
            # batch_times[tid] continues increasing.
            # So Wave 2 starts after Wave 1 finishes?
            # Correct (due to temp reuse).
            # Wait. If tid 0 finishes early, can tid 0 of wave 2 start?
            # Yes. `batch_times[tid]` is the "Free at" time.
            # So it fits perfectly.
            pass

        self.instrs.extend(scheduler.get_instrs())

        # Epilogue
        store_loop = []
        for b in range(0, batch_size, VLEN):
            b_const = self.scratch_const(b)
            addr_reg = self.alloc_scratch("addr")
            store_loop.append({"alu": [("+", addr_reg, inp_indices_p_addr, b_const)]})
            store_loop.append({"store": [("vstore", addr_reg, self.IDX_BASE + b)]})
            store_loop.append({"alu": [("+", addr_reg, inp_values_p_addr, b_const)]})
            store_loop.append({"store": [("vstore", addr_reg, self.VAL_BASE + b)]})
        self.instrs.extend(store_loop)
        self.instrs.append({"flow": [("pause",)]})

    def gen_batch_all_rounds(self, b_off, rounds, hash_consts, forest_base, TWO, ZERO, ONE, N_NODES, t_base):
        
        T_NODE_VEC = t_base
        T_ADDR_VEC = t_base + 8
        T_TMP1 = t_base + 16
        T_TMP2 = t_base + 24
        
        idx_vec = self.IDX_BASE + b_off
        val_vec = self.VAL_BASE + b_off
        
        for r in range(rounds):
            # Step 1: Addr Calc
            ops = []
            for i in range(VLEN):
                ops.append(Op("alu", ("+", T_ADDR_VEC + i, forest_base, idx_vec + i)))
            yield ops
            
            # Step 2: Load
            ops = []
            for i in range(VLEN):
                 ops.append(Op("load", ("load", T_NODE_VEC + i, T_ADDR_VEC + i)))
            yield ops
            
            # Step 3: Hash
            yield [Op("valu", ("^", val_vec, val_vec, T_NODE_VEC))]
            for hi, (c1, c3) in enumerate(hash_consts):
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                yield [Op("valu", (op1, T_TMP1, val_vec, c1))]
                yield [Op("valu", (op3, T_TMP2, val_vec, c3))]
                yield [Op("valu", (op2, val_vec, T_TMP1, T_TMP2))]

                
            # Step 4: Update
            yield [Op("valu", ("%", T_TMP1, val_vec, TWO))]
            yield [Op("valu", ("==", T_TMP1, T_TMP1, ZERO))]
            yield [Op("flow", ("vselect", T_TMP2, T_TMP1, ONE, TWO))]
            yield [Op("valu", ("*", idx_vec, idx_vec, TWO))]
            yield [Op("valu", ("+", idx_vec, idx_vec, T_TMP2))]
            yield [Op("valu", ("<", T_TMP1, idx_vec, N_NODES))]
            yield [Op("flow", ("vselect", idx_vec, T_TMP1, idx_vec, ZERO))]
 


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = True,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)



    kb = OptimizedKernelBuilder()
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
    machine.run()
    
    # Run reference kernel to completion to get final state
    ref_mem = None
    for ref_mem in reference_kernel2(mem, value_trace):
        pass
    
    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect result values"
    
    inp_indices_p = ref_mem[5]
    if prints:
        print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        
    assert (
        machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
        == ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]
    ), "Incorrect result indices"

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
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256, trace=False, prints=False)


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
