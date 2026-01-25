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

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def pack_vliw(self, slot_groups: list[list[tuple[str, tuple]]]):
        """
        Pack pre-grouped slots into VLIW instruction bundles.
        Each group in slot_groups represents slots that can execute in parallel.
        """
        instrs = []
        for group in slot_groups:
            bundle = defaultdict(list)
            for engine, slot in group:
                bundle[engine].append(slot)
            if bundle:
                instrs.append(dict(bundle))
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

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash_vector(self, val_vec, tmp1_vec, tmp2_vec, const_vecs):
        """Build vectorized hash computation for VLEN elements in parallel."""
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("valu", (op1, tmp1_vec, val_vec, const_vecs[val1])))
            slots.append(("valu", (op3, tmp2_vec, val_vec, const_vecs[val3])))
            slots.append(("valu", (op2, val_vec, tmp1_vec, tmp2_vec)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        12-way parallelism with 6+6 balanced split for reduced register pressure:
        Fewer chunks allow all XOR and hash combine ops to fit in single cycles.
        Each wave has exactly 6 chunks matching the 6 VALU slot limit.
        This may improve instruction-level parallelism.
        """
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

        hash_consts = set()
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            hash_consts.add(val1)
            hash_consts.add(val3)

        const_vecs = {}
        for val in hash_consts:
            vec_addr = self.alloc_scratch(f"const_vec_{val}", VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            const_vecs[val] = vec_addr

        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)

        self.add("valu", ("vbroadcast", zero_vec, zero_const))
        self.add("valu", ("vbroadcast", one_vec, one_const))
        self.add("valu", ("vbroadcast", two_vec, two_const))
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "12-way parallelism with 6+6 balanced split"))

        num_vector_chunks = batch_size // VLEN
        NUM_PARALLEL = 12
        WAVE1_SIZE = 6

        chunk_regs = []
        for c in range(NUM_PARALLEL):
            regs = {
                'vec_idx': self.alloc_scratch(f"vec_idx_{c}", VLEN),
                'vec_val': self.alloc_scratch(f"vec_val_{c}", VLEN),
                'vec_node_val': self.alloc_scratch(f"vec_node_val_{c}", VLEN),
                'vec_tmp1': self.alloc_scratch(f"vec_tmp1_{c}", VLEN),
                'vec_tmp2': self.alloc_scratch(f"vec_tmp2_{c}", VLEN),
                'vec_cond': self.alloc_scratch(f"vec_cond_{c}", VLEN),
                'addr_idx': self.alloc_scratch(f"addr_idx_{c}"),
                'addr_val': self.alloc_scratch(f"addr_val_{c}"),
                'node_addrs': [self.alloc_scratch(f"node_addr_{c}_{i}") for i in range(VLEN)],
            }
            chunk_regs.append(regs)

        num_groups = (num_vector_chunks + NUM_PARALLEL - 1) // NUM_PARALLEL
        for group_idx in range(num_groups):
            active_chunks = []
            for i in range(NUM_PARALLEL):
                chunk_num = group_idx * NUM_PARALLEL + i
                if chunk_num < num_vector_chunks:
                    active_chunks.append((i, chunk_num))

            wave1 = active_chunks[:WAVE1_SIZE]
            wave2 = active_chunks[WAVE1_SIZE:]

            base_consts = {}
            for c_idx, chunk_num in active_chunks:
                base_offset = chunk_num * VLEN
                base_consts[c_idx] = self.scratch_const(base_offset)

            alu_ops = []
            for c_idx, chunk_num in active_chunks:
                r = chunk_regs[c_idx]
                alu_ops.append(("+", r['addr_idx'], self.scratch["inp_indices_p"], base_consts[c_idx]))
                alu_ops.append(("+", r['addr_val'], self.scratch["inp_values_p"], base_consts[c_idx]))
            while alu_ops:
                batch, alu_ops = alu_ops[:12], alu_ops[12:]
                self.instrs.append({"alu": batch})

            for pair_start in range(0, len(active_chunks), 2):
                load_ops = []
                for p in range(2):
                    if pair_start + p < len(active_chunks):
                        c_idx, _ = active_chunks[pair_start + p]
                        r = chunk_regs[c_idx]
                        load_ops.append(("vload", r['vec_idx'], r['addr_idx']))
                self.instrs.append({"load": load_ops})
                load_ops = []
                for p in range(2):
                    if pair_start + p < len(active_chunks):
                        c_idx, _ = active_chunks[pair_start + p]
                        r = chunk_regs[c_idx]
                        load_ops.append(("vload", r['vec_val'], r['addr_val']))
                self.instrs.append({"load": load_ops})

            for round_num in range(rounds):
                is_first_round = (round_num == 0)
                is_last_round = (round_num == rounds - 1)

                if not wave2:
                    all_alu_ops = []
                    for c_idx, _ in active_chunks:
                        r = chunk_regs[c_idx]
                        for i in range(VLEN):
                            all_alu_ops.append(("+", r['node_addrs'][i], self.scratch["forest_values_p"], r['vec_idx'] + i))
                    while all_alu_ops:
                        batch, all_alu_ops = all_alu_ops[:12], all_alu_ops[12:]
                        self.instrs.append({"alu": batch})

                    all_loads = []
                    for c_idx, _ in active_chunks:
                        r = chunk_regs[c_idx]
                        for i in range(VLEN):
                            all_loads.append(("load", r['vec_node_val'] + i, r['node_addrs'][i]))
                    while all_loads:
                        batch, all_loads = all_loads[:2], all_loads[2:]
                        self.instrs.append({"load": batch})

                    xor_ops = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        xor_ops.append(("^", r['vec_val'], r['vec_val'], r['vec_node_val']))
                    while xor_ops:
                        batch, xor_ops = xor_ops[:6], xor_ops[6:]
                        self.instrs.append({"valu": batch})

                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        fused_ops = []
                        for c_idx, _ in wave1:
                            r = chunk_regs[c_idx]
                            fused_ops.append((op1, r['vec_tmp1'], r['vec_val'], const_vecs[val1]))
                            fused_ops.append((op3, r['vec_tmp2'], r['vec_val'], const_vecs[val3]))
                        while fused_ops:
                            batch, fused_ops = fused_ops[:6], fused_ops[6:]
                            self.instrs.append({"valu": batch})

                        combine_ops = []
                        for c_idx, _ in wave1:
                            r = chunk_regs[c_idx]
                            combine_ops.append((op2, r['vec_val'], r['vec_tmp1'], r['vec_tmp2']))
                        while combine_ops:
                            batch, combine_ops = combine_ops[:6], combine_ops[6:]
                            self.instrs.append({"valu": batch})

                else:
                    wave1_alu_ops = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        for i in range(VLEN):
                            wave1_alu_ops.append(("+", r['node_addrs'][i], self.scratch["forest_values_p"], r['vec_idx'] + i))
                    while wave1_alu_ops:
                        batch, wave1_alu_ops = wave1_alu_ops[:12], wave1_alu_ops[12:]
                        self.instrs.append({"alu": batch})

                    wave1_loads = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        for i in range(VLEN):
                            wave1_loads.append(("load", r['vec_node_val'] + i, r['node_addrs'][i]))

                    wave2_alu_ops = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        for i in range(VLEN):
                            wave2_alu_ops.append(("+", r['node_addrs'][i], self.scratch["forest_values_p"], r['vec_idx'] + i))

                    wave1_load_idx = 0
                    while wave2_alu_ops:
                        bundle = {}
                        alu_batch, wave2_alu_ops = wave2_alu_ops[:12], wave2_alu_ops[12:]
                        bundle["alu"] = alu_batch
                        if wave1_load_idx < len(wave1_loads):
                            load_batch = wave1_loads[wave1_load_idx:wave1_load_idx+2]
                            wave1_load_idx += 2
                            bundle["load"] = load_batch
                        self.instrs.append(bundle)

                    while wave1_load_idx < len(wave1_loads):
                        load_batch = wave1_loads[wave1_load_idx:wave1_load_idx+2]
                        wave1_load_idx += 2
                        self.instrs.append({"load": load_batch})

                    wave2_loads = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        for i in range(VLEN):
                            wave2_loads.append(("load", r['vec_node_val'] + i, r['node_addrs'][i]))

                    xor_ops = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        xor_ops.append(("^", r['vec_val'], r['vec_val'], r['vec_node_val']))

                    wave2_load_idx = 0
                    while xor_ops:
                        batch, xor_ops = xor_ops[:6], xor_ops[6:]
                        bundle = {"valu": batch}
                        if wave2_load_idx < len(wave2_loads):
                            load_batch = wave2_loads[wave2_load_idx:wave2_load_idx+2]
                            wave2_load_idx += 2
                            bundle["load"] = load_batch
                        self.instrs.append(bundle)

                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        fused_ops = []
                        for c_idx, _ in wave1:
                            r = chunk_regs[c_idx]
                            fused_ops.append((op1, r['vec_tmp1'], r['vec_val'], const_vecs[val1]))
                            fused_ops.append((op3, r['vec_tmp2'], r['vec_val'], const_vecs[val3]))

                        while fused_ops:
                            batch, fused_ops = fused_ops[:6], fused_ops[6:]
                            bundle = {"valu": batch}
                            if wave2_load_idx < len(wave2_loads):
                                load_batch = wave2_loads[wave2_load_idx:wave2_load_idx+2]
                                wave2_load_idx += 2
                                bundle["load"] = load_batch
                            self.instrs.append(bundle)

                        combine_ops = []
                        for c_idx, _ in wave1:
                            r = chunk_regs[c_idx]
                            combine_ops.append((op2, r['vec_val'], r['vec_tmp1'], r['vec_tmp2']))

                        while combine_ops:
                            batch, combine_ops = combine_ops[:6], combine_ops[6:]
                            bundle = {"valu": batch}
                            if wave2_load_idx < len(wave2_loads):
                                load_batch = wave2_loads[wave2_load_idx:wave2_load_idx+2]
                                wave2_load_idx += 2
                                bundle["load"] = load_batch
                            self.instrs.append(bundle)

                    while wave2_load_idx < len(wave2_loads):
                        load_batch = wave2_loads[wave2_load_idx:wave2_load_idx+2]
                        wave2_load_idx += 2
                        self.instrs.append({"load": load_batch})

                if wave2:
                    w1_post = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        w1_post.append(("&", r['vec_tmp1'], r['vec_val'], one_vec))
                    w2_xor = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        w2_xor.append(("^", r['vec_val'], r['vec_val'], r['vec_node_val']))
                    combined = w1_post + w2_xor
                    while combined:
                        batch, combined = combined[:6], combined[6:]
                        self.instrs.append({"valu": batch})

                    w1_post = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        w1_post.append(("+", r['vec_tmp2'], r['vec_tmp1'], one_vec))
                    op1, val1, op2, op3, val3 = HASH_STAGES[0]
                    w2_hash = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        w2_hash.append((op1, r['vec_tmp1'], r['vec_val'], const_vecs[val1]))
                        w2_hash.append((op3, r['vec_tmp2'], r['vec_val'], const_vecs[val3]))
                    combined = w1_post + w2_hash
                    while combined:
                        batch, combined = combined[:6], combined[6:]
                        self.instrs.append({"valu": batch})

                    w1_post = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        w1_post.append(("multiply_add", r['vec_idx'], r['vec_idx'], two_vec, r['vec_tmp2']))
                    w2_hash = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        w2_hash.append((op2, r['vec_val'], r['vec_tmp1'], r['vec_tmp2']))
                    combined = w1_post + w2_hash
                    while combined:
                        batch, combined = combined[:6], combined[6:]
                        self.instrs.append({"valu": batch})

                    w1_compare = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        w1_compare.append(("<", r['vec_cond'], r['vec_idx'], n_nodes_vec))

                    w1_stores = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        w1_stores.append(("vstore", r['addr_val'], r['vec_val']))

                    w1_store_idx = 0
                    for hi in range(1, len(HASH_STAGES)):
                        op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                        fused_ops = []
                        for c_idx, _ in wave2:
                            r = chunk_regs[c_idx]
                            fused_ops.append((op1, r['vec_tmp1'], r['vec_val'], const_vecs[val1]))
                            fused_ops.append((op3, r['vec_tmp2'], r['vec_val'], const_vecs[val3]))

                        if hi == 1:
                            fused_ops = w1_compare + fused_ops
                        while fused_ops:
                            batch, fused_ops = fused_ops[:6], fused_ops[6:]
                            bundle = {"valu": batch}
                            if w1_store_idx < len(w1_stores):
                                bundle["store"] = w1_stores[w1_store_idx:w1_store_idx+2]
                                w1_store_idx += 2
                            self.instrs.append(bundle)

                        valu_batch = []
                        for c_idx, _ in wave2:
                            r = chunk_regs[c_idx]
                            valu_batch.append((op2, r['vec_val'], r['vec_tmp1'], r['vec_tmp2']))
                        if hi == 1:
                            w1_multiply = []
                            for c_idx, _ in wave1:
                                r = chunk_regs[c_idx]
                                w1_multiply.append(("*", r['vec_idx'], r['vec_idx'], r['vec_cond']))
                            valu_batch = w1_multiply + valu_batch
                        while valu_batch:
                            batch, valu_batch = valu_batch[:6], valu_batch[6:]
                            bundle = {"valu": batch}
                            if w1_store_idx < len(w1_stores):
                                bundle["store"] = w1_stores[w1_store_idx:w1_store_idx+2]
                                w1_store_idx += 2
                            self.instrs.append(bundle)

                    while w1_store_idx < len(w1_stores):
                        self.instrs.append({"store": w1_stores[w1_store_idx:w1_store_idx+2]})
                        w1_store_idx += 2

                    w2_val_stores = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        w2_val_stores.append(("vstore", r['addr_val'], r['vec_val']))

                    w2_post = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        w2_post.append(("&", r['vec_tmp1'], r['vec_val'], one_vec))
                    while w2_post:
                        batch, w2_post = w2_post[:6], w2_post[6:]
                        bundle = {"valu": batch}
                        if w2_val_stores:
                            bundle["store"] = w2_val_stores[:2]
                            w2_val_stores = w2_val_stores[2:]
                        self.instrs.append(bundle)

                    w2_post = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        w2_post.append(("+", r['vec_tmp2'], r['vec_tmp1'], one_vec))
                    while w2_post:
                        batch, w2_post = w2_post[:6], w2_post[6:]
                        bundle = {"valu": batch}
                        if w2_val_stores:
                            bundle["store"] = w2_val_stores[:2]
                            w2_val_stores = w2_val_stores[2:]
                        self.instrs.append(bundle)

                    combined_post = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        combined_post.append(("multiply_add", r['vec_idx'], r['vec_idx'], two_vec, r['vec_tmp2']))
                    while combined_post:
                        batch, combined_post = combined_post[:6], combined_post[6:]
                        bundle = {"valu": batch}
                        if w2_val_stores:
                            bundle["store"] = w2_val_stores[:2]
                            w2_val_stores = w2_val_stores[2:]
                        self.instrs.append(bundle)

                    combined_post = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        combined_post.append(("<", r['vec_cond'], r['vec_idx'], n_nodes_vec))
                    while combined_post:
                        batch, combined_post = combined_post[:6], combined_post[6:]
                        bundle = {"valu": batch}
                        if w2_val_stores:
                            bundle["store"] = w2_val_stores[:2]
                            w2_val_stores = w2_val_stores[2:]
                        self.instrs.append(bundle)

                    combined_post = []
                    for c_idx, _ in wave2:
                        r = chunk_regs[c_idx]
                        combined_post.append(("*", r['vec_idx'], r['vec_idx'], r['vec_cond']))
                    while combined_post:
                        batch, combined_post = combined_post[:6], combined_post[6:]
                        bundle = {"valu": batch}
                        if w2_val_stores:
                            bundle["store"] = w2_val_stores[:2]
                            w2_val_stores = w2_val_stores[2:]
                        self.instrs.append(bundle)

                    while w2_val_stores:
                        self.instrs.append({"store": w2_val_stores[:2]})
                        w2_val_stores = w2_val_stores[2:]

                else:
                    combined_post = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        combined_post.append(("&", r['vec_tmp1'], r['vec_val'], one_vec))
                    while combined_post:
                        batch, combined_post = combined_post[:6], combined_post[6:]
                        self.instrs.append({"valu": batch})

                    combined_post = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        combined_post.append(("+", r['vec_tmp2'], r['vec_tmp1'], one_vec))
                    while combined_post:
                        batch, combined_post = combined_post[:6], combined_post[6:]
                        self.instrs.append({"valu": batch})

                    combined_post = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        combined_post.append(("multiply_add", r['vec_idx'], r['vec_idx'], two_vec, r['vec_tmp2']))
                    while combined_post:
                        batch, combined_post = combined_post[:6], combined_post[6:]
                        self.instrs.append({"valu": batch})

                    combined_post = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        combined_post.append(("<", r['vec_cond'], r['vec_idx'], n_nodes_vec))
                    while combined_post:
                        batch, combined_post = combined_post[:6], combined_post[6:]
                        self.instrs.append({"valu": batch})

                    combined_post = []
                    for c_idx, _ in wave1:
                        r = chunk_regs[c_idx]
                        combined_post.append(("*", r['vec_idx'], r['vec_idx'], r['vec_cond']))
                    while combined_post:
                        batch, combined_post = combined_post[:6], combined_post[6:]
                        self.instrs.append({"valu": batch})

                    all_stores = []
                    for c_idx, _ in active_chunks:
                        r = chunk_regs[c_idx]
                        all_stores.append(("vstore", r['addr_val'], r['vec_val']))
                    while all_stores:
                        batch, all_stores = all_stores[:2], all_stores[2:]
                        self.instrs.append({"store": batch})

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
