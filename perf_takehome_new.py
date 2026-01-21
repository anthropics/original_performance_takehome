"""
FIRST PRINCIPLES LOOP-BASED KERNEL
Trace back to (0,0) - minimal instruction count with runtime loops
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class LoopKernelBuilder(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        LOOP-BASED kernel - process ONE vector group at a time with nested loops.
        Generates ~100 instructions that execute (rounds * groups) times via jumps.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Initialize parameters from memory
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        # Loop counters
        round_counter = self.alloc_scratch("round_counter")
        group_counter = self.alloc_scratch("group_counter")

        # Working registers for ONE vector group
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node = self.alloc_scratch("v_node", VLEN)
        v_tmp = self.alloc_scratch("v_tmp", VLEN)
        hash_v1 = self.alloc_scratch("hash_v1", VLEN)
        hash_v2 = self.alloc_scratch("hash_v2", VLEN)

        # Vector constants
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.emit(valu=[
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ])

        # Pre-broadcast hash constants
        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1 = self.alloc_scratch(f"vc1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"vc3_{hi}", VLEN)
            self.emit(valu=[
                ("vbroadcast", v_c1, self.scratch_const(val1)),
                ("vbroadcast", v_c3, self.scratch_const(val3)),
            ])
            hash_consts.append((v_c1, v_c3))

        # Addressing temps
        tmp_addr = self.alloc_scratch("tmp_addr")
        node_addrs = [self.alloc_scratch(f"node_addr{i}") for i in range(VLEN)]
        group_offset = self.alloc_scratch("group_offset")

        # Initialize counters
        self.emit(load=[("const", round_counter, 0)])
        self.emit(load=[("const", group_counter, 0)])

        # ============ OUTER LOOP START (rounds) ============
        round_loop_start = len(self.instrs)

        # Reset group counter at start of each round
        self.emit(load=[("const", group_counter, 0)])

        # ============ INNER LOOP START (groups) ============
        group_loop_start = len(self.instrs)

        # Calculate group_offset = group_counter * 8
        self.emit(alu=[("<<", group_offset, group_counter, self.scratch_const(3))])  # shift left 3 = *8

        # Load indices
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], group_offset)])
        self.emit(load=[("vload", v_idx, tmp_addr)])

        # Load values
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_values_p"], group_offset)])
        self.emit(load=[("vload", v_val, tmp_addr)])

        # Calculate node addresses and load (8 scalar operations)
        for vi in range(VLEN):
            self.emit(alu=[("+", node_addrs[vi], self.scratch["forest_values_p"], v_idx + vi)])
        for vi in range(VLEN):
            self.emit(load=[("load", v_node + vi, node_addrs[vi])])

        # XOR
        self.emit(valu=[("^", v_val, v_val, v_node)])

        # Hash (6 stages, 3 ops per stage = 18 valu ops)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = hash_consts[hi]
            self.emit(valu=[
                (op1, hash_v1, v_val, v_c1),
                (op3, hash_v2, v_val, v_c3),
            ])
            self.emit(valu=[(op2, v_val, hash_v1, hash_v2)])

        # Index calculation
        self.emit(valu=[("%", v_tmp, v_val, v_two)])
        self.emit(valu=[("+", v_tmp, v_one, v_tmp)])
        self.emit(valu=[("*", v_idx, v_idx, v_two)])
        self.emit(valu=[("+", v_idx, v_idx, v_tmp)])

        # Bounds check
        self.emit(valu=[("<", v_tmp, v_idx, v_n_nodes)])
        self.emit(valu=[("*", v_idx, v_idx, v_tmp)])

        # Store results
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], group_offset)])
        self.emit(store=[("vstore", tmp_addr, v_idx)])
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_values_p"], group_offset)])
        self.emit(store=[("vstore", tmp_addr, v_val)])

        # Increment group_counter
        self.emit(flow=[("add_imm", group_counter, group_counter, 1)])

        # Jump back if group_counter < 32
        self.emit(alu=[("<", tmp1, group_counter, self.scratch_const(32))])
        self.emit(flow=[("cond_jump", tmp1, group_loop_start)])

        # Increment round_counter
        self.emit(flow=[("add_imm", round_counter, round_counter, 1)])

        # Jump back if round_counter < rounds
        self.emit(alu=[("<", tmp1, round_counter, self.scratch["rounds"])])
        self.emit(flow=[("cond_jump", tmp1, round_loop_start)])

        # Done
        self.instrs.append({"flow": [("pause",)]})

# Test it
if __name__ == "__main__":
    from tests.submission_tests import do_kernel_test

    # Monkey-patch KernelBuilder to use loop version
    import perf_takehome
    original = perf_takehome.KernelBuilder.build_kernel
    perf_takehome.KernelBuilder.build_kernel = LoopKernelBuilder.build_kernel

    result = do_kernel_test(10, 16, 256)
    print(f"Loop-based kernel: {result} cycles")
