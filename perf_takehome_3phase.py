"""
3-PHASE RESTRUCTURED KERNEL
Separate I/O from compute for better packing and cache behavior.

Phase 1: LOAD all indices/values into scratch (ONCE)
Phase 2: COMPUTE all rounds on all groups (with node loads)
Phase 3: STORE all indices/values from scratch (ONCE)
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class ThreePhaseKernel(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """Three clear phases: LOAD → COMPUTE → STORE"""
        tmp1 = self.alloc_scratch("tmp1")

        # Initialize
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

        # ========== QUANTUM STATE: ALL indices and values in scratch ==========
        all_idx = [self.alloc_scratch(f"idx{g}", VLEN) for g in range(n_groups)]
        all_val = [self.alloc_scratch(f"val{g}", VLEN) for g in range(n_groups)]

        # Working registers
        v_node = self.alloc_scratch("v_node", VLEN)
        v_tmp = self.alloc_scratch("v_tmp", VLEN)
        hash_v1 = self.alloc_scratch("hash_v1", VLEN)
        hash_v2 = self.alloc_scratch("hash_v2", VLEN)

        # Constants
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
        tmp_addr = self.alloc_scratch("tmp_addr")
        node_addrs = [self.alloc_scratch(f"na{i}") for i in range(VLEN)]

        # ========== PHASE 1: LOAD ALL (Quantum state initialization) ==========
        for g in range(n_groups):
            # Pack loads tightly
            self.emit(flow=[("add_imm", tmp_addr, self.scratch["inp_indices_p"], g * VLEN)])
            self.emit(load=[("vload", all_idx[g], tmp_addr)])
            self.emit(flow=[("add_imm", tmp_addr, self.scratch["inp_values_p"], g * VLEN)])
            self.emit(load=[("vload", all_val[g], tmp_addr)])

        # ========== PHASE 2: COMPUTE ALL (Transform quantum state) ==========
        for r in range(rounds):
            for g in range(n_groups):
                # Calculate node addresses (pack into 2-4 bundles)
                for offset in range(0, VLEN, 4):
                    ops = [("+", node_addrs[offset+i], self.scratch["forest_values_p"], all_idx[g] + offset + i)
                           for i in range(min(4, VLEN-offset))]
                    self.emit(alu=ops)

                # Load nodes (pack 2 per bundle)
                for offset in range(0, VLEN, 2):
                    self.emit(load=[
                        ("load", v_node + offset, node_addrs[offset]),
                        ("load", v_node + offset + 1, node_addrs[offset + 1]),
                    ])

                # XOR
                self.emit(valu=[("^", all_val[g], all_val[g], v_node)])

                # Hash (packed)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v_c1, v_c3 = hash_consts[hi]
                    self.emit(valu=[
                        (op1, hash_v1, all_val[g], v_c1),
                        (op3, hash_v2, all_val[g], v_c3),
                    ])
                    self.emit(valu=[(op2, all_val[g], hash_v1, hash_v2)])

                # Index calculation (packed)
                self.emit(valu=[
                    ("%", v_tmp, all_val[g], v_two),
                    ("+", v_tmp, v_one, v_tmp),
                ])
                self.emit(valu=[
                    ("*", all_idx[g], all_idx[g], v_two),
                    ("+", all_idx[g], all_idx[g], v_tmp),
                ])

                # Bounds check (packed)
                self.emit(valu=[
                    ("<", v_tmp, all_idx[g], v_n_nodes),
                    ("*", all_idx[g], all_idx[g], v_tmp),
                ])

        # ========== PHASE 3: STORE ALL (Collapse quantum state) ==========
        for g in range(n_groups):
            self.emit(flow=[("add_imm", tmp_addr, self.scratch["inp_indices_p"], g * VLEN)])
            self.emit(store=[("vstore", tmp_addr, all_idx[g])])
            self.emit(flow=[("add_imm", tmp_addr, self.scratch["inp_values_p"], g * VLEN)])
            self.emit(store=[("vstore", tmp_addr, all_val[g])])

        # Done
        self.instrs.append({"flow": [("pause",)]})

# Test both sides
if __name__ == "__main__":
    import perf_takehome

    print("="*70)
    print("TESTING: 3-PHASE RESTRUCTURED KERNEL")
    print("="*70)

    original_build = perf_takehome.KernelBuilder.build_kernel
    perf_takehome.KernelBuilder.build_kernel = ThreePhaseKernel.build_kernel

    from perf_takehome import do_kernel_test
    cycles_3phase = do_kernel_test(10, 16, 256)

    print(f"\n3-Phase Kernel: {cycles_3phase} cycles")
    print(f"Current Best: 5,028 cycles")
    print(f"Target: 1,487 cycles")
    print("="*70)

    if cycles_3phase < 5028:
        improvement = 5028 - cycles_3phase
        print(f"✓✓✓ BREAKTHROUGH! Improved by {improvement} cycles ({100*improvement/5028:.1f}%)")
        if cycles_3phase < 1487:
            print(f"✓✓✓✓✓ SURPASSED OPUS 4.5 TARGET!")
    else:
        print(f"Not a breakthrough: {cycles_3phase} vs 5,028")

    print("="*70)
