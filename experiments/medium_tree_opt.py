
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    class Scheduler:
        def __init__(self, parent):
            self.parent = parent
            self.streams = [] # List of lists of (engine, slot)
            self.current_ptrs = []
            
        def add_stream(self, ops):
            self.streams.append(ops)
            self.current_ptrs.append(0)
            
        def emit_all(self):
            # Greedy packing
            active = True
            while active:
                bundle = defaultdict(list)
                slots_used = defaultdict(int)
                progress = False
                
                # Check all streams
                for i in range(len(self.streams)):
                    ptr = self.current_ptrs[i]
                    stream = self.streams[i]
                    
                    if ptr < len(stream):
                        engine, slot = stream[ptr]
                        limit = SLOT_LIMITS[engine]
                        
                        if slots_used[engine] < limit:
                            bundle[engine].append(slot)
                            slots_used[engine] += 1
                            self.current_ptrs[i] += 1
                            progress = True
                            
                if progress:
                    self.parent.emit(bundle)
                else:
                    if all(self.current_ptrs[i] >= len(self.streams[i]) for i in range(len(self.streams))):
                        active = False
                    else:
                        pass

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def emit(self, bundle):
        self.instrs.append(bundle)

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
            slots.append(
                ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
            )
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

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
        three_const = self.scratch_const(3) # OPT

        hash_consts = []
        fusable_stages = set()
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                fusable_stages.add(hi)
                factor = 1 + (1 << val3)
                c_factor = self.alloc_scratch(f"hc_factor_{hi}", VLEN)
                c_const = self.alloc_scratch(f"hc_const_{hi}", VLEN)
                self.emit({"load": [("const", tmp1, factor), ("const", tmp2, val1)]})
                self.emit(
                    {
                        "valu": [
                            ("vbroadcast", c_factor, tmp1),
                            ("vbroadcast", c_const, tmp2),
                        ]
                    }
                )
                hash_consts.append((c_factor, c_const, True))
            else:
                c1 = self.alloc_scratch(f"hc1_{hi}", VLEN)
                c3 = self.alloc_scratch(f"hc3_{hi}", VLEN)
                self.emit({"load": [("const", tmp1, val1), ("const", tmp2, val3)]})
                self.emit(
                    {"valu": [("vbroadcast", c1, tmp1), ("vbroadcast", c3, tmp2)]}
                )
                hash_consts.append((c1, c3, False))

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_three = self.alloc_scratch("v_three", VLEN) # OPT
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        self.emit(
            {
                "valu": [
                    ("vbroadcast", v_zero, zero_const),
                    ("vbroadcast", v_one, one_const),
                    ("vbroadcast", v_two, two_const),
                    ("vbroadcast", v_three, three_const), # OPT
                    ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
                    ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
                ]
            }
        )

        n_vectors = batch_size // VLEN
        s_idx = self.alloc_scratch("s_idx", n_vectors * VLEN)
        s_val = self.alloc_scratch("s_val", n_vectors * VLEN)

        BATCH = 16
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(BATCH)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(BATCH)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(BATCH)]
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(BATCH)]
        v_node = [self.alloc_scratch(f"v_node_{i}", VLEN) for i in range(BATCH)]
        v_addr = [self.alloc_scratch(f"v_addr_{i}", VLEN) for i in range(BATCH)]
        
        v_broadcast_node = self.alloc_scratch("v_broadcast_node", VLEN)
        addr_tmp = self.alloc_scratch("addr_tmp")

        v_tree_1 = self.alloc_scratch("v_tree_1", VLEN)
        v_tree_2 = self.alloc_scratch("v_tree_2", VLEN)
        v_tree_diff = self.alloc_scratch("v_tree_diff", VLEN)
        
        # New constants for Medium Tree (Users additions)
        v_tree_3 = self.alloc_scratch("v_tree_3", VLEN)
        v_tree_4 = self.alloc_scratch("v_tree_4", VLEN)
        v_tree_5 = self.alloc_scratch("v_tree_5", VLEN)
        v_tree_6 = self.alloc_scratch("v_tree_6", VLEN)
        v_tree_diff34 = self.alloc_scratch("v_tree_diff34", VLEN)
        v_tree_diff56 = self.alloc_scratch("v_tree_diff56", VLEN)
        
        # Load Optimization Constants (Tree 1,2)
        self.emit({"load": [("const", tmp1, 1), ("const", tmp2, 2)]})
        self.emit(
            {
                "alu": [
                    ("+", tmp1, self.scratch["forest_values_p"], tmp1),
                    ("+", tmp2, self.scratch["forest_values_p"], tmp2),
                ]
            }
        )
        self.emit({"load": [("load", tmp1, tmp1), ("load", tmp2, tmp2)]})
        self.emit(
            {"valu": [("vbroadcast", v_tree_1, tmp1), ("vbroadcast", v_tree_2, tmp2)]}
        )
        self.emit({"valu": [("-", v_tree_diff, v_tree_2, v_tree_1)]})
        
        # Load Optimization Constants (Tree 3,4)
        self.emit({"load": [("const", tmp1, 3), ("const", tmp2, 4)]})
        self.emit(
            {
                "alu": [
                    ("+", tmp1, self.scratch["forest_values_p"], tmp1),
                    ("+", tmp2, self.scratch["forest_values_p"], tmp2),
                ]
            }
        )
        self.emit({"load": [("load", tmp1, tmp1), ("load", tmp2, tmp2)]})
        self.emit(
            {"valu": [("vbroadcast", v_tree_3, tmp1), ("vbroadcast", v_tree_4, tmp2)]}
        )
        self.emit({"valu": [("-", v_tree_diff34, v_tree_3, v_tree_4)]}) # T3 - T4
        
        # Load Optimization Constants (Tree 5,6)
        self.emit({"load": [("const", tmp1, 5), ("const", tmp2, 6)]})
        self.emit(
            {
                "alu": [
                    ("+", tmp1, self.scratch["forest_values_p"], tmp1),
                    ("+", tmp2, self.scratch["forest_values_p"], tmp2),
                ]
            }
        )
        self.emit({"load": [("load", tmp1, tmp1), ("load", tmp2, tmp2)]})
        self.emit(
            {"valu": [("vbroadcast", v_tree_5, tmp1), ("vbroadcast", v_tree_6, tmp2)]}
        )
        self.emit({"valu": [("-", v_tree_diff56, v_tree_5, v_tree_6)]}) # T5 - T6

        self.add("flow", ("pause",))

        for vi in range(n_vectors):
            off = vi * VLEN
            self.emit({"load": [("const", addr_tmp, off)]})
            self.emit(
                {"alu": [("+", addr_tmp, self.scratch["inp_indices_p"], addr_tmp)]}
            )
            self.emit({"load": [("vload", s_idx + off, addr_tmp)]})

        for vi in range(n_vectors):
            off = vi * VLEN
            self.emit({"load": [("const", addr_tmp, off)]})
            self.emit(
                {"alu": [("+", addr_tmp, self.scratch["inp_values_p"], addr_tmp)]}
            )
            self.emit({"load": [("vload", s_val + off, addr_tmp)]})

        broadcast_rounds = {0, 11}

        for rnd in range(rounds):
            is_broadcast_round = rnd in broadcast_rounds
            
            if is_broadcast_round:
                self.emit({"load": [("load", tmp1, self.scratch["forest_values_p"])]})
                self.emit({"valu": [("vbroadcast", v_broadcast_node, tmp1)]})

            for batch_start in range(0, n_vectors, BATCH):
                batch_end = min(batch_start + BATCH, n_vectors)
                cur_batch_size = batch_end - batch_start
                offs = [(batch_start + i) * VLEN for i in range(cur_batch_size)]
                
                sched = self.Scheduler(self)
                sched.streams = [[] for _ in range(cur_batch_size)]
                sched.current_ptrs = [0] * cur_batch_size
                
                for i in range(cur_batch_size):
                    stream = []
                    
                    if is_broadcast_round:
                         stream.append(("valu", ("^", v_val[i], s_val + offs[i], v_broadcast_node)))
                         
                    elif rnd in {1, 12}: # Small Tree Optimization (Ids 0..2)
                         stream.append(("valu", ("+", v_idx[i], s_idx + offs[i], v_zero)))
                         stream.append(("valu", ("&", v_tmp1[i], v_idx[i], v_one)))
                         stream.append(("valu", ("^", v_tmp1[i], v_tmp1[i], v_one)))
                         stream.append(("valu", ("multiply_add", v_node[i], v_tmp1[i], v_tree_diff, v_tree_1)))
                         stream.append(("valu", ("+", v_val[i], s_val + offs[i], v_zero)))
                         stream.append(("valu", ("^", v_val[i], v_val[i], v_node[i])))
                         
                    elif rnd in {2, 13}: # Medium Tree Optimization (Ids 0..6)
                        # Load idx
                        stream.append(("valu", ("+", v_idx[i], s_idx + offs[i], v_zero))) 
                        
                        # bit0 = idx & 1
                        stream.append(("valu", ("&", v_tmp1[i], v_idx[i], v_one)))
                        
                        # T34 = T4 + bit0*(T3-T4). If 1->T3. If 0->T4.
                        stream.append(("valu", ("multiply_add", v_tmp2[i], v_tmp1[i], v_tree_diff34, v_tree_4)))
                        
                        # T56 = T6 + bit0*(T5-T6). If 1->T5. If 0->T6.
                        stream.append(("valu", ("multiply_add", v_node[i], v_tmp1[i], v_tree_diff56, v_tree_6)))
                        
                        # Parent Selection.
                        # parent = (idx - 1) >> 1.
                        # cond = parent & 1. (1 if P1->T34, 0 if P2->T56).
                        
                        stream.append(("valu", ("-", v_tmp1[i], v_idx[i], v_one))) # idx - 1
                        stream.append(("valu", (">>", v_tmp1[i], v_tmp1[i], v_one))) # (idx-1) >> 1
                        stream.append(("valu", ("&", v_tmp1[i], v_tmp1[i], v_one))) # cond
                        
                        # Result = T56 + cond * (T34 - T56)
                        stream.append(("valu", ("-", v_node[i], v_tmp2[i], v_node[i]))) # diff = T34 - T56. (Wait. v_tmp2 is T34. v_node is T56.)
                        # Usually dest overrides source? 
                        # "-" op: dest, src1, src2. v_node = v_tmp2 - v_node.
                        # wait. v_node (T56) is needed as Base?
                        # formula: Base + cond * (Target - Base).
                        # Base=T56. Target=T34.
                        # Diff = T34 - T56.
                        # We use v_node as dest for diff?
                        # Then we lose Base.
                        # We need Base for the final multiply_add.
                        # registers: v_tmp2(T34), v_node(T56). v_tmp1(cond).
                        # We need T56 later.
                        # So calculate Diff into v_tmp2?
                        # v_tmp2 = v_tmp2 - v_node. (T34 - T56).
                        # Now v_tmp2 is Diff. v_node is Base. v_tmp1 is Cond.
                        # Result = v_tmp1 * v_tmp2 + v_node.
                        # multiply_add(dest, src1, src2, src3). dest = src1*src2 + src3.
                        # multiply_add(v_node, v_tmp1, v_tmp2, v_node).
                        # CORRECT.
                        
                        stream.append(("valu", ("-", v_tmp2[i], v_tmp2[i], v_node[i]))) # T34 - T56
                        stream.append(("valu", ("multiply_add", v_node[i], v_tmp1[i], v_tmp2[i], v_node[i])))
                        
                        stream.append(("valu", ("+", v_val[i], s_val + offs[i], v_zero)))
                        stream.append(("valu", ("^", v_val[i], v_val[i], v_node[i])))
                         
                    else: # Gather Logic
                         stream.append(("valu", ("+", v_idx[i], s_idx + offs[i], v_zero)))
                         stream.append(("valu", ("+", v_addr[i], v_forest_p, v_idx[i])))
                         for j in range(0, VLEN, 2):
                             stream.append(("load", ("load_offset", v_node[i], v_addr[i], j)))
                             stream.append(("load", ("load_offset", v_node[i], v_addr[i], j+1)))
                         stream.append(("valu", ("+", v_val[i], s_val + offs[i], v_zero)))
                         stream.append(("valu", ("^", v_val[i], v_val[i], v_node[i])))
                    
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                            hc = hash_consts[hi]
                            if hc[2]: 
                                c_factor, c_const = hc[0], hc[1]
                                stream.append(("valu", ("multiply_add", v_val[i], v_val[i], c_factor, c_const)))
                            else:
                                c1, c3 = hc[0], hc[1]
                                stream.append(("valu", (op1, v_tmp1[i], v_val[i], c1)))
                                stream.append(("valu", (op3, v_tmp2[i], v_val[i], c3)))
                                stream.append(("valu", (op2, v_val[i], v_tmp1[i], v_tmp2[i])))
                    
                    stream.append(("valu", ("&", v_tmp1[i], v_val[i], v_one)))
                    stream.append(("valu", ("+", v_tmp2[i], v_one, v_tmp1[i])))
                    stream.append(("valu", ("multiply_add", v_idx[i], s_idx + offs[i], v_two, v_tmp2[i])))
                    stream.append(("valu", ("<", v_tmp1[i], v_idx[i], v_n_nodes)))
                    stream.append(("valu", ("*", v_idx[i], v_idx[i], v_tmp1[i])))
                    stream.append(("valu", ("+", s_idx + offs[i], v_idx[i], v_zero)))
                    stream.append(("valu", ("+", s_val + offs[i], v_val[i], v_zero)))
                    
                    sched.streams[i] = stream
                
                sched.emit_all()

        for vi in range(n_vectors):
            off = vi * VLEN
            self.emit({"load": [("const", addr_tmp, off)]})
            self.emit(
                {"alu": [("+", addr_tmp, self.scratch["inp_indices_p"], addr_tmp)]}
            )
            self.emit({"store": [("vstore", addr_tmp, s_idx + off)]})

        for vi in range(n_vectors):
            off = vi * VLEN
            self.emit({"load": [("const", addr_tmp, off)]})
            self.emit(
                {"alu": [("+", addr_tmp, self.scratch["inp_values_p"], addr_tmp)]}
            )
            self.emit({"store": [("vstore", addr_tmp, s_val + off)]})

        self.emit({"flow": [("pause",)]})


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


if __name__ == "__main__":
    unittest.main()
