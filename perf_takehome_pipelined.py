"""
PIPELINED KERNEL - Overlap I/O with Computation
Fill the "trailing negative space" by overlapping:
- Store batch N while loading batch N+1
- Eliminate idle cycles at batch boundaries
"""

from perf_takehome import KernelBuilder, HASH_STAGES, VLEN

class PipelinedKernel(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Pipeline loads/stores with computation.
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

        # DOUBLE BUFFERING: Two sets of registers
        # While processing buffer A, load into buffer B
        v_idx_a = [self.alloc_scratch(f"v_idx_a{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_val_a = [self.alloc_scratch(f"v_val_a{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_idx_b = [self.alloc_scratch(f"v_idx_b{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_val_b = [self.alloc_scratch(f"v_val_b{g}", VLEN) for g in range(PARALLEL_GROUPS)]

        v_node = [self.alloc_scratch(f"v_node{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        v_tmp = [self.alloc_scratch(f"v_tmp{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        hash_v1 = [self.alloc_scratch(f"hash_v1_{g}", VLEN) for g in range(PARALLEL_GROUPS)]
        hash_v2 = [self.alloc_scratch(f"hash_v2_{g}", VLEN) for g in range(PARALLEL_GROUPS)]

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
        tmp_addr_a = [self.alloc_scratch(f"tmp_addr_a{g}") for g in range(PARALLEL_GROUPS)]
        tmp_addr_b = [self.alloc_scratch(f"tmp_addr_b{g}") for g in range(PARALLEL_GROUPS)]
        node_addr = [[self.alloc_scratch(f"node{g}_{v}") for v in range(VLEN)] for g in range(PARALLEL_GROUPS)]

        # ========== LOAD FIRST BATCH ==========
        batch_start = 0
        for bg in range(PARALLEL_GROUPS):
            g = batch_start + bg
            self.emit(flow=[("add_imm", tmp_addr_a[bg], self.scratch["inp_indices_p"], g * VLEN)])
        for bg in range(PARALLEL_GROUPS):
            self.emit(load=[("vload", v_idx_a[bg], tmp_addr_a[bg])])
        for bg in range(PARALLEL_GROUPS):
            g = batch_start + bg
            self.emit(flow=[("add_imm", tmp_addr_a[bg], self.scratch["inp_values_p"], g * VLEN)])
        for bg in range(PARALLEL_GROUPS):
            self.emit(load=[("vload", v_val_a[bg], tmp_addr_a[bg])])

        # Process batches with pipelining
        for batch_idx in range(n_groups // PARALLEL_GROUPS):
            batch_start = batch_idx * PARALLEL_GROUPS
            batch_size_actual = PARALLEL_GROUPS

            # Select which buffer to use (alternate A/B)
            if batch_idx % 2 == 0:
                v_idx = v_idx_a
                v_val = v_val_a
                tmp_addr = tmp_addr_a
                v_idx_next = v_idx_b
                v_val_next = v_val_b
                tmp_addr_next = tmp_addr_b
            else:
                v_idx = v_idx_b
                v_val = v_val_b
                tmp_addr = tmp_addr_b
                v_idx_next = v_idx_a
                v_val_next = v_val_a
                tmp_addr_next = tmp_addr_a

            # If not last batch, start loading next batch IN PARALLEL with computation
            next_batch_start = (batch_idx + 1) * PARALLEL_GROUPS
            has_next = next_batch_start < n_groups

            # ========== PROCESS CURRENT BATCH THROUGH ALL ROUNDS ==========
            for round_i in range(rounds):
                # Calculate addresses (12 ALU ops per bundle)
                for offset in range(0, batch_size_actual * VLEN, 12):
                    alu_ops = []
                    for i in range(min(12, batch_size_actual * VLEN - offset)):
                        bg = (offset + i) // VLEN
                        vi = (offset + i) % VLEN
                        if bg < batch_size_actual:
                            alu_ops.append(("+", node_addr[bg][vi], self.scratch["forest_values_p"], v_idx[bg] + vi))

                    # PIPELINE: If first round and has next batch, compute next batch addresses while doing this
                    if round_i == 0 and has_next and len(alu_ops) < 12:
                        # Add address calculations for next batch load
                        for bg in range(min(2, PARALLEL_GROUPS)):  # Just first 2 to fit in 12 slots
                            g = next_batch_start + bg
                            if g < n_groups:
                                alu_ops.append(("add_imm", tmp_addr_next[bg], self.scratch["inp_indices_p"], g * VLEN))

                    if alu_ops:
                        self.emit(alu=alu_ops)

                # Load nodes (2 loads per bundle)
                for offset in range(0, batch_size_actual * VLEN, 2):
                    load_ops = []
                    for i in range(min(2, batch_size_actual * VLEN - offset)):
                        bg = (offset + i) // VLEN
                        vi = (offset + i) % VLEN
                        if bg < batch_size_actual:
                            load_ops.append(("load", v_node[bg] + vi, node_addr[bg][vi]))

                    # PIPELINE: If round 0 and has next, start loading next batch
                    if round_i == 0 and has_next and offset < 4:  # First few load bundles
                        bg = offset // 2
                        if bg < PARALLEL_GROUPS:
                            load_ops.append(("vload", v_idx_next[bg], tmp_addr_next[bg]))

                    if load_ops:
                        self.emit(load=load_ops)

                # XOR, Hash, Index calc - same as before (all parallel)
                xor_ops = [("^", v_val[bg], v_val[bg], v_node[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=xor_ops)

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v_c1, v_c3 = hash_consts[hi]
                    ops1_3 = []
                    for bg in range(batch_size_actual):
                        ops1_3.append((op1, hash_v1[bg], v_val[bg], v_c1))
                    for bg in range(batch_size_actual):
                        ops1_3.append((op3, hash_v2[bg], v_val[bg], v_c3))
                    for offset in range(0, len(ops1_3), 6):
                        self.emit(valu=ops1_3[offset:offset+6])
                    ops2 = [(op2, v_val[bg], hash_v1[bg], hash_v2[bg]) for bg in range(batch_size_actual)]
                    self.emit(valu=ops2)

                mod_ops = [("%", v_tmp[bg], v_val[bg], v_two) for bg in range(batch_size_actual)]
                self.emit(valu=mod_ops)
                add_ops = [("+", v_tmp[bg], v_one, v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=add_ops)
                mul_ops = [("*", v_idx[bg], v_idx[bg], v_two) for bg in range(batch_size_actual)]
                self.emit(valu=mul_ops)
                add_ops2 = [("+", v_idx[bg], v_idx[bg], v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=add_ops2)

                cmp_ops = [("<", v_tmp[bg], v_idx[bg], v_n_nodes) for bg in range(batch_size_actual)]
                self.emit(valu=cmp_ops)
                mul_ops2 = [("*", v_idx[bg], v_idx[bg], v_tmp[bg]) for bg in range(batch_size_actual)]
                self.emit(valu=mul_ops2)

            # ========== STORE CURRENT BATCH ==========
            for bg in range(batch_size_actual):
                g = batch_start + bg
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_indices_p"], g * VLEN)])
            for bg in range(batch_size_actual):
                self.emit(store=[("vstore", tmp_addr[bg], v_idx[bg])])
            for bg in range(batch_size_actual):
                g = batch_start + bg
                self.emit(flow=[("add_imm", tmp_addr[bg], self.scratch["inp_values_p"], g * VLEN)])
            for bg in range(batch_size_actual):
                self.emit(store=[("vstore", tmp_addr[bg], v_val[bg])])

        # Done
        self.instrs.append({"flow": [("pause",)]})


if __name__ == "__main__":
    import perf_takehome
    from perf_takehome import do_kernel_test

    print("="*70)
    print("PIPELINED KERNEL - Overlapped I/O")
    print("="*70)

    original_build = perf_takehome.KernelBuilder.build_kernel
    perf_takehome.KernelBuilder.build_kernel = PipelinedKernel.build_kernel

    cycles = do_kernel_test(10, 16, 256)

    print(f"\n{'='*70}")
    print(f"Pipelined Kernel: {cycles} cycles")
    print(f"Previous Best: 4,997 cycles")
    print(f"Target: 1,790 cycles (Opus 4.5 casual)")
    print(f"{'='*70}")

    if cycles < 4997:
        improvement = 4997 - cycles
        print(f"✓✓✓ IMPROVEMENT: {improvement} cycles ({100*improvement/4997:.1f}%)")
        if cycles < 1790:
            print(f"✓✓✓✓✓ BEAT OPUS 4.5 CASUAL TARGET!")
            print(f"🎯 SUBMIT THIS!")
    else:
        print(f"No improvement - pipelining overhead > benefit")

    print(f"{'='*70}")
