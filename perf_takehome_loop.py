"""
LOOP-BASED implementation - First Principles approach
Uses jump instructions to create tight inner loops
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class LoopKernelBuilder(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Tight loop processing 8 elements (1 vector) per iteration.
        Inner loop: 32 iterations (batch_size / VLEN)
        Outer loop: 16 rounds
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Initialize
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        eight_const = self.scratch_const(8)

        self.add("flow", ("pause",))

        # Allocate loop counters
        round_counter = self.alloc_scratch("round_counter")
        group_counter = self.alloc_scratch("group_counter")
        base_offset = self.alloc_scratch("base_offset")

        # Vector registers (only need 1 set since we loop)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node = self.alloc_scratch("v_node", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
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

        # Scalar temps
        tmp_addr = self.alloc_scratch("tmp_addr")
        node_addrs = [self.alloc_scratch(f"node_addr{i}") for i in range(VLEN)]

        # Initialize round_counter = 0
        self.emit(load=[("const", round_counter, 0)])

        # OUTER LOOP START (rounds)
        round_loop_start = len(self.instrs)

        # Initialize group_counter = 0
        self.emit(load=[("const", group_counter, 0)])

        # INNER LOOP START (groups)
        group_loop_start = len(self.instrs)

        # Calculate base_offset = group_counter * 8
        self.emit(alu=[("<<", base_offset, group_counter, self.scratch_const(3))])  # shift left 3 = * 8

        # Load indices vector
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], base_offset)])
        self.emit(load=[("vload", v_idx, tmp_addr)])

        # Load values vector
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_values_p"], base_offset)])
        self.emit(load=[("vload", v_val, tmp_addr)])

        # Calculate node addresses
        for vi in range(VLEN):
            self.emit(alu=[("+", node_addrs[vi], self.scratch["forest_values_p"], v_idx + vi)])

        # Load node values
        for vi in range(VLEN):
            self.emit(load=[("load", v_node + vi, node_addrs[vi])])

        # XOR
        self.emit(valu=[("^", v_val, v_val, v_node)])

        # Hash (6 stages)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = hash_consts[hi]
            self.emit(valu=[
                (op1, hash_v1, v_val, v_c1),
                (op3, hash_v2, v_val, v_c3),
            ])
            self.emit(valu=[(op2, v_val, hash_v1, hash_v2)])

        # Index calculation
        self.emit(valu=[
            ("%", v_tmp1, v_val, v_two),
            ("+", v_tmp1, v_one, v_tmp1),
        ])
        self.emit(valu=[
            ("*", v_idx, v_idx, v_two),
            ("+", v_idx, v_idx, v_tmp1),
        ])

        # Bounds check
        self.emit(valu=[
            ("<", v_tmp1, v_idx, v_n_nodes),
            ("*", v_idx, v_idx, v_tmp1),
        ])

        # Store results
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], base_offset)])
        self.emit(store=[("vstore", tmp_addr, v_idx)])
        self.emit(alu=[("+", tmp_addr, self.scratch["inp_values_p"], base_offset)])
        self.emit(store=[("vstore", tmp_addr, v_val)])

        # Increment group_counter
        self.emit(flow=[("add_imm", group_counter, group_counter, 1)])

        # Check if group_counter < 32, if so jump back
        self.emit(alu=[("<", tmp1, group_counter, self.scratch_const(32))])
        self.emit(flow=[("cond_jump", tmp1, group_loop_start)])

        # Increment round_counter
        self.emit(flow=[("add_imm", round_counter, round_counter, 1)])

        # Check if round_counter < rounds, if so jump back
        self.emit(alu=[("<", tmp1, round_counter, self.scratch["rounds"])])
        self.emit(flow=[("cond_jump", tmp1, round_loop_start)])

        # Done
        self.instrs.append({"flow": [("pause",)]})
