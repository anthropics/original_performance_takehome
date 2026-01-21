"""
GOLDEN BUNDLE PACKING + VSELECT PREPARATION
Get back to 5K cycles with proper packing, then add vselect deduplication
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class GoldenVSelectKernel(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Apply golden bundle packing like the working kernel.
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Initialize parameters
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

        n_groups = batch_size // VLEN  # 32 groups
        PARALLEL_GROUPS = 6

        # Working registers (matching working kernel structure)
        v_idx = [self.alloc_scratch(f"v_idx{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_val = [self.alloc_scratch(f"v_val{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_node = self.alloc_scratch("v_node", VLEN)

        # Index calculation temps
        v_idx_tmp1 = self.alloc_scratch("v_idx_tmp1", VLEN)
        v_idx_tmp2 = self.alloc_scratch("v_idx_tmp2", VLEN)
        v_idx_tmp3 = self.alloc_scratch("v_idx_tmp3", VLEN)
        v_bounds_check = self.alloc_scratch("v_bounds_check", VLEN)

        # Hash working registers
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

        # Hash constants
        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1 = self.alloc_scratch(f"vc1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"vc3_{hi}", VLEN)
            self.emit(valu=[
                ("vbroadcast", v_c1, self.scratch_const(val1)),
                ("vbroadcast", v_c3, self.scratch_const(val3)),
            ])
            hash_consts.append((v_c1, v_c3))

        # Addressing
        tmp_addr = [self.alloc_scratch(f"tmp_addr{g}") for g in range(PARALLEL_GROUPS)]
        node_addrs = [self.alloc_scratch(f"node_addr{i}") for i in range(VLEN)]

        # Process in batches for register reuse (TIME amortization)
        for batch_start in range(0, n_groups, PARALLEL_GROUPS):
            batch_size_actual = min(PARALLEL_GROUPS, n_groups - batch_start)

            # PHASE 1: LOAD indices and values (ONCE per batch)
            for bg in range(batch_size_actual):
                g = batch_start + bg
                offset = g * VLEN
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_indices_p"], offset)])
                self.emit(load=[("vload", v_idx[bg], tmp_addr[bg])])
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_values_p"], offset)])
                self.emit(load=[("vload", v_val[bg], tmp_addr[bg])])

            # PHASE 2: Process ALL rounds with register reuse
            for round_i in range(rounds):
                for bg in range(batch_size_actual):
                    # GOLDEN BUNDLE PACKING: Pack address calculations (12 ALU ops per bundle)
                    total_addrs = batch_size_actual * VLEN
                    for offset in range(0, total_addrs, 12):
                        alu_ops = []
                        for i in range(min(12, total_addrs - offset)):
                            curr_bg = offset // VLEN
                            curr_vi = offset % VLEN
                            if curr_bg < batch_size_actual:
                                alu_ops.append(("+", node_addrs[curr_vi],
                                              self.scratch["forest_values_p"],
                                              v_idx[curr_bg] + curr_vi))
                        if alu_ops:
                            self.emit(alu=alu_ops)

                    # Calculate node addresses for this specific bg
                    for vi in range(0, VLEN, 12):
                        ops = [("+", node_addrs[vi+i], self.scratch["forest_values_p"], v_idx[bg] + vi + i)
                               for i in range(min(12, VLEN-vi))]
                        self.emit(alu=ops)

                    # GOLDEN BUNDLE PACKING: Pack loads (2 per bundle)
                    for vi in range(0, VLEN, 2):
                        self.emit(load=[
                            ("load", v_node + vi, node_addrs[vi]),
                            ("load", v_node + vi + 1, node_addrs[vi + 1]),
                        ])

                    # XOR
                    self.emit(valu=[("^", v_val[bg], v_val[bg], v_node)])

                    # Hash with interleaving (matching working kernel)
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        v_c1, v_c3 = hash_consts[hi]
                        self.emit(valu=[
                            (op1, hash_v1, v_val[bg], v_c1),
                            (op3, hash_v2, v_val[bg], v_c3),
                        ])
                        self.emit(valu=[(op2, v_val[bg], hash_v1, hash_v2)])

                    # Index calculation (properly split to avoid hazards)
                    self.emit(valu=[("%", v_idx_tmp1, v_val[bg], v_two)])
                    self.emit(valu=[
                        ("+", v_idx_tmp3, v_one, v_idx_tmp1),
                        ("*", v_idx_tmp2, v_idx[bg], v_two),
                    ])
                    self.emit(valu=[("+", v_idx[bg], v_idx_tmp2, v_idx_tmp3)])

                    # Bounds check
                    self.emit(valu=[("<", v_bounds_check, v_idx[bg], v_n_nodes)])
                    self.emit(valu=[("*", v_idx[bg], v_idx[bg], v_bounds_check)])

            # PHASE 3: STORE results (ONCE per batch)
            for bg in range(batch_size_actual):
                g = batch_start + bg
                offset = g * VLEN
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_indices_p"], offset)])
                self.emit(store=[("vstore", tmp_addr[bg], v_idx[bg])])
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_values_p"], offset)])
                self.emit(store=[("vstore", tmp_addr[bg], v_val[bg])])

        # Done
        self.instrs.append({"flow": [("pause",)]})


# Test it
if __name__ == "__main__":
    import perf_takehome

    print("="*70)
    print("TESTING: GOLDEN BUNDLE PACKING + VSELECT PREP")
    print("="*70)

    original_build = perf_takehome.KernelBuilder.build_kernel
    perf_takehome.KernelBuilder.build_kernel = GoldenVSelectKernel.build_kernel

    from perf_takehome import do_kernel_test
    cycles = do_kernel_test(10, 16, 256)

    print(f"\nGolden VSelect Kernel: {cycles} cycles")
    print(f"Current Best: 5,028 cycles")
    print(f"Target: 1,487 cycles")
    print("="*70)

    if cycles <= 5028:
        print(f"✓ Back to baseline or better!")
        if cycles < 5028:
            improvement = 5028 - cycles
            print(f"✓ IMPROVEMENT: {improvement} cycles better ({100*improvement/5028:.1f}%)")
    else:
        print(f"Need more packing: {cycles} vs 5,028")

    print("="*70)
