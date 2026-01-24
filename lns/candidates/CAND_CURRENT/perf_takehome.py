"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

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
        instrs = []
        for item in slots:
            if isinstance(item, dict):
                instrs.append(item)
            else:
                engine, slot = item
                if engine == "hash":
                    val_hash_addr, tmp1, tmp2, round, i = slot
                    instrs.extend(self.build_hash(val_hash_addr, tmp1, tmp2, round, i))
                else:
                    instrs.append({engine: [slot]})
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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        instrs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            instrs.append({"alu": [(op1, tmp1, val_hash_addr, self.scratch_const(val1)),
                                   (op3, tmp2, val_hash_addr, self.scratch_const(val3))]})
            instrs.append({"alu": [(op2, val_hash_addr, tmp1, tmp2)],
                           "debug": [("compare", val_hash_addr, (round, i, "hash_stage", hi))]})
        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Instruction-interleaved implementation: interleaves independent operations
        across different chunks to hide latency and maximize VLIW utilization.
        """
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

        tmp_addrs = [self.alloc_scratch(f"tmp_addr_{i}") for i in range(len(init_vars))]
        load_const_instrs = []
        for i in range(0, len(init_vars), 2):
            loads = [("const", tmp_addrs[i], i)]
            if i + 1 < len(init_vars):
                loads.append(("const", tmp_addrs[i+1], i+1))
            load_const_instrs.append({"load": loads})
        self.instrs.extend(load_const_instrs)

        load_var_instrs = []
        for i in range(0, len(init_vars), 2):
            loads = [("load", self.scratch[init_vars[i]], tmp_addrs[i])]
            if i + 1 < len(init_vars):
                loads.append(("load", self.scratch[init_vars[i+1]], tmp_addrs[i+1]))
            load_var_instrs.append({"load": loads})
        self.instrs.extend(load_var_instrs)

        one_const = self.scratch_const(1)
        zero_const = self.scratch_const(0)

        self.add("flow", ("pause",))

        body = []

        n_chunks = batch_size // VLEN

        chunk_data = self.alloc_scratch("chunk_data", n_chunks * 2 * VLEN)
        idx_chunks = [chunk_data + c * 2 * VLEN for c in range(n_chunks)]
        val_chunks = [chunk_data + c * 2 * VLEN + VLEN for c in range(n_chunks)]

        node_val_vecs = []
        for c in range(n_chunks):
            node_val_vecs.append(self.alloc_scratch(f"node_val_vec_{c}", VLEN))

        addr_vecs = []
        for c in range(n_chunks):
            addr_vecs.append(self.alloc_scratch(f"addr_vec_{c}", VLEN))

        addr_tmp = self.alloc_scratch("addr_tmp")
        addr_tmp2 = self.alloc_scratch("addr_tmp2")

        n_temps = min(6, n_chunks)
        bounds_masks = [self.alloc_scratch(f"bounds_mask_{i}", VLEN) for i in range(n_temps)]
        hash_t1s = [self.alloc_scratch(f"hash_t1_{i}", VLEN) for i in range(n_temps)]
        hash_t2s = [self.alloc_scratch(f"hash_t2_{i}", VLEN) for i in range(n_temps)]

        one_vec = self.alloc_scratch("one_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        forest_p_vec = self.alloc_scratch("forest_p_vec", VLEN)

        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val1_const = self.scratch_const(val1)
            val3_const = self.scratch_const(val3)
            val1_vec = self.alloc_scratch(f"hc1_{hi}", VLEN)
            val3_vec = self.alloc_scratch(f"hc3_{hi}", VLEN)
            hash_const_vecs.append((val1_vec, val3_vec, val1_const, val3_const))

        mul_4097_const = self.scratch_const(4097)
        mul_33_const = self.scratch_const(33)
        mul_9_const = self.scratch_const(9)
        mul_2_const = self.scratch_const(2)
        mul_4097_vec = self.alloc_scratch("mul_4097", VLEN)
        mul_33_vec = self.alloc_scratch("mul_33", VLEN)
        mul_9_vec = self.alloc_scratch("mul_9", VLEN)
        mul_2_vec = self.alloc_scratch("mul_2", VLEN)
        mul_vecs = {0: mul_4097_vec, 2: mul_33_vec, 4: mul_9_vec}

        i_const_0 = self.scratch_const(0)
        val1_vec_0, val3_vec_0, val1_const_0, val3_const_0 = hash_const_vecs[0]
        val1_vec_1, val3_vec_1, val1_const_1, val3_const_1 = hash_const_vecs[1]
        val1_vec_2, val3_vec_2, val1_const_2, val3_const_2 = hash_const_vecs[2]
        val1_vec_3, val3_vec_3, val1_const_3, val3_const_3 = hash_const_vecs[3]
        val1_vec_4, val3_vec_4, val1_const_4, val3_const_4 = hash_const_vecs[4]
        val1_vec_5, val3_vec_5, val1_const_5, val3_const_5 = hash_const_vecs[5]
        body.append({"valu": [("vbroadcast", one_vec, one_const),
                              ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]),
                              ("vbroadcast", zero_vec, zero_const),
                              ("vbroadcast", forest_p_vec, self.scratch["forest_values_p"]),
                              ("vbroadcast", val1_vec_0, val1_const_0),
                              ("vbroadcast", mul_4097_vec, mul_4097_const)],
                     "alu": [("+", addr_tmp, self.scratch["inp_indices_p"], i_const_0),
                             ("+", addr_tmp2, self.scratch["inp_values_p"], i_const_0)]})

        body.append({"valu": [("vbroadcast", val1_vec_1, val1_const_1),
                              ("vbroadcast", val3_vec_1, val3_const_1),
                              ("vbroadcast", val1_vec_2, val1_const_2),
                              ("vbroadcast", val1_vec_3, val1_const_3),
                              ("vbroadcast", val3_vec_3, val3_const_3),
                              ("vbroadcast", mul_33_vec, mul_33_const)]})

        chunk_offsets = [self.scratch_const(c * VLEN) for c in range(n_chunks + 1)]

        body.append({"load": [("vload", idx_chunks[0], addr_tmp),
                              ("vload", val_chunks[0], addr_tmp2)],
                     "alu": [("+", addr_tmp, self.scratch["inp_indices_p"], chunk_offsets[1]),
                             ("+", addr_tmp2, self.scratch["inp_values_p"], chunk_offsets[1])],
                     "valu": [("vbroadcast", val1_vec_4, val1_const_4),
                              ("vbroadcast", val1_vec_5, val1_const_5),
                              ("vbroadcast", val3_vec_5, val3_const_5),
                              ("vbroadcast", mul_9_vec, mul_9_const),
                              ("vbroadcast", mul_2_vec, mul_2_const)]})

        for c in range(1, n_chunks):
            next_chunk = c + 1
            next_i_const = chunk_offsets[next_chunk] if next_chunk <= n_chunks else chunk_offsets[0]
            body.append({"load": [("vload", idx_chunks[c], addr_tmp),
                                  ("vload", val_chunks[c], addr_tmp2)],
                         "alu": [("+", addr_tmp, self.scratch["inp_indices_p"], next_i_const),
                                 ("+", addr_tmp2, self.scratch["inp_values_p"], next_i_const)]})

        for round_num in range(rounds):
            for c in range(0, n_chunks, 6):
                ops = []
                for cc in range(c, min(c + 6, n_chunks)):
                    ops.append(("+", addr_vecs[cc], forest_p_vec, idx_chunks[cc]))
                body.append({"valu": ops})

            xor_queue = []
            for c in range(n_chunks):
                addr_vec = addr_vecs[c]
                node_val_vec = node_val_vecs[c]
                if c > 0:
                    xor_queue.append(("^", val_chunks[c-1], val_chunks[c-1], node_val_vecs[c-1]))
                xor_batch = xor_queue[:6]
                xor_queue = xor_queue[6:]
                if xor_batch:
                    body.append({"load": [("load", node_val_vec, addr_vec),
                                          ("load", node_val_vec + 1, addr_vec + 1)],
                                 "valu": xor_batch})
                else:
                    body.append({"load": [("load", node_val_vec, addr_vec),
                                          ("load", node_val_vec + 1, addr_vec + 1)]})
                body.append({"load": [("load", node_val_vec + 2, addr_vec + 2),
                                      ("load", node_val_vec + 3, addr_vec + 3)]})
                body.append({"load": [("load", node_val_vec + 4, addr_vec + 4),
                                      ("load", node_val_vec + 5, addr_vec + 5)]})
                body.append({"load": [("load", node_val_vec + 6, addr_vec + 6),
                                      ("load", node_val_vec + 7, addr_vec + 7)]})
            xor_queue.append(("^", val_chunks[n_chunks-1], val_chunks[n_chunks-1], node_val_vecs[n_chunks-1]))
            while xor_queue:
                xor_batch = xor_queue[:6]
                xor_queue = xor_queue[6:]
                body.append({"valu": xor_batch})

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                val1_vec, val3_vec, _, _ = hash_const_vecs[hi]
                if hi in mul_vecs:
                    mul_vec = mul_vecs[hi]
                    for c in range(0, n_chunks, 6):
                        ma_ops = []
                        for cc in range(c, min(c + 6, n_chunks)):
                            ma_ops.append(("multiply_add", val_chunks[cc], val_chunks[cc], mul_vec, val1_vec))
                        body.append({"valu": ma_ops})
                else:
                    for g in range(0, n_chunks, 6):
                        group_end = min(g + 6, n_chunks)
                        for c in range(g, group_end, 3):
                            step1_ops = []
                            for cc in range(c, min(c + 3, group_end)):
                                tmp_idx = cc - g
                                step1_ops.append((op1, hash_t1s[tmp_idx], val_chunks[cc], val1_vec))
                                step1_ops.append((op3, hash_t2s[tmp_idx], val_chunks[cc], val3_vec))
                            body.append({"valu": step1_ops})
                        step2_ops = []
                        for cc in range(g, group_end):
                            tmp_idx = cc - g
                            step2_ops.append((op2, val_chunks[cc], hash_t1s[tmp_idx], hash_t2s[tmp_idx]))
                        body.append({"valu": step2_ops})

            for g in range(0, n_chunks, 6):
                group_end = min(g + 6, n_chunks)
                group_size = group_end - g
                and_ops = []
                ma_ops = []
                for cc in range(g, group_end):
                    tmp_idx = cc - g
                    and_ops.append(("&", hash_t2s[tmp_idx], val_chunks[cc], one_vec))
                    ma_ops.append(("multiply_add", idx_chunks[cc], idx_chunks[cc], mul_2_vec, one_vec))
                if group_size <= 3:
                    body.append({"valu": and_ops + ma_ops})
                else:
                    body.append({"valu": and_ops})
                    body.append({"valu": ma_ops})

                add_ops = []
                for cc in range(g, group_end):
                    tmp_idx = cc - g
                    add_ops.append(("+", idx_chunks[cc], idx_chunks[cc], hash_t2s[tmp_idx]))
                body.append({"valu": add_ops})

                cmp_ops = []
                for cc in range(g, group_end):
                    tmp_idx = cc - g
                    cmp_ops.append(("<", bounds_masks[tmp_idx], idx_chunks[cc], n_nodes_vec))
                body.append({"valu": cmp_ops})

                mul_ops = []
                for cc in range(g, group_end):
                    tmp_idx = cc - g
                    mul_ops.append(("*", idx_chunks[cc], idx_chunks[cc], bounds_masks[tmp_idx]))
                if round_num == rounds - 1 and g == 0:
                    body.append({"valu": mul_ops,
                                 "alu": [("+", addr_tmp, self.scratch["inp_indices_p"], i_const_0),
                                         ("+", addr_tmp2, self.scratch["inp_values_p"], i_const_0)]})
                else:
                    body.append({"valu": mul_ops})

        next_i_const_0 = self.scratch_const(VLEN)
        body.append({"store": [("vstore", addr_tmp, idx_chunks[0]),
                               ("vstore", addr_tmp2, val_chunks[0])],
                     "alu": [("+", addr_tmp, self.scratch["inp_indices_p"], next_i_const_0),
                             ("+", addr_tmp2, self.scratch["inp_values_p"], next_i_const_0)]})

        for c in range(1, n_chunks - 1):
            next_i = (c + 1) * VLEN
            next_i_const = self.scratch_const(next_i)
            body.append({"store": [("vstore", addr_tmp, idx_chunks[c]),
                                   ("vstore", addr_tmp2, val_chunks[c])],
                         "alu": [("+", addr_tmp, self.scratch["inp_indices_p"], next_i_const),
                                 ("+", addr_tmp2, self.scratch["inp_values_p"], next_i_const)]})
        body.append({"store": [("vstore", addr_tmp, idx_chunks[n_chunks-1]),
                               ("vstore", addr_tmp2, val_chunks[n_chunks-1])],
                     "flow": [("pause",)]})

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)

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
