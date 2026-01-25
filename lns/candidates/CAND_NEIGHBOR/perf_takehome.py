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
        Broadcast coalescing optimization: In early rounds, all batch elements
        share the same forest index (all start at 0). After round k, there are
        at most 2^k distinct indices. For rounds where indices are guaranteed
        identical across all elements, we load once and broadcast instead of
        doing 8 scalar loads per chunk. This saves ~4 cycles per chunk for
        early rounds.
        """
        needed_vars = [
            ("n_nodes", 1),
            ("forest_values_p", 4),
            ("inp_indices_p", 5),
            ("inp_values_p", 6),
        ]
        for v, _ in needed_vars:
            self.alloc_scratch(v, 1)

        tmp_addrs = [self.alloc_scratch(f"tmp_addr_{i}") for i in range(len(needed_vars) - 1)]
        one_const = self.alloc_scratch()
        zero_const = self.alloc_scratch()
        self.instrs.append({"load": [("const", one_const, 1), ("const", tmp_addrs[0], 4)]})
        self.instrs.append({"load": [("const", tmp_addrs[1], 5), ("const", tmp_addrs[2], 6)]})
        self.instrs.append({"load": [("load", self.scratch["n_nodes"], one_const),
                                      ("load", self.scratch["forest_values_p"], tmp_addrs[0])]})
        self.instrs.append({"load": [("load", self.scratch["inp_indices_p"], tmp_addrs[1]),
                                      ("load", self.scratch["inp_values_p"], tmp_addrs[2])]})

        self.instrs.append({"load": [("const", zero_const, 0)], "flow": [("pause",)]})
        self.const_map[1] = one_const
        self.const_map[0] = zero_const

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
        broadcast_tmp = self.alloc_scratch("broadcast_tmp")

        n_temps = min(6, n_chunks)
        bounds_masks = [self.alloc_scratch(f"bounds_mask_{i}", VLEN) for i in range(n_temps)]
        hash_t1s = [self.alloc_scratch(f"hash_t1_{i}", VLEN) for i in range(n_temps)]
        hash_t2s = [self.alloc_scratch(f"hash_t2_{i}", VLEN) for i in range(n_temps)]

        one_vec = self.alloc_scratch("one_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        forest_p_vec = self.alloc_scratch("forest_p_vec", VLEN)

        hash_const_vecs = []
        hash_val1_consts = []
        hash_val3_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val1_const = self.alloc_scratch()
            hash_val1_consts.append((val1_const, val1))
            if val3 not in self.const_map:
                val3_const = self.alloc_scratch()
                hash_val3_consts.append((val3_const, val3))
                self.const_map[val3] = val3_const
            else:
                val3_const = self.const_map[val3]
            val1_vec = self.alloc_scratch(f"hc1_{hi}", VLEN)
            val3_vec = self.alloc_scratch(f"hc3_{hi}", VLEN)
            hash_const_vecs.append((val1_vec, val3_vec, val1_const, val3_const))
        for i in range(0, len(hash_val1_consts), 2):
            loads = [("const", hash_val1_consts[i][0], hash_val1_consts[i][1])]
            if i + 1 < len(hash_val1_consts):
                loads.append(("const", hash_val1_consts[i+1][0], hash_val1_consts[i+1][1]))
            self.instrs.append({"load": loads})
        for addr, val in hash_val1_consts:
            self.const_map[val] = addr
        for i in range(0, len(hash_val3_consts), 2):
            loads = [("const", hash_val3_consts[i][0], hash_val3_consts[i][1])]
            if i + 1 < len(hash_val3_consts):
                loads.append(("const", hash_val3_consts[i+1][0], hash_val3_consts[i+1][1]))
            self.instrs.append({"load": loads})

        mul_4097_const = self.alloc_scratch()
        mul_33_const = self.alloc_scratch()
        mul_2_const = self.alloc_scratch()
        vlen_const = self.alloc_scratch()
        mul_9_const = self.alloc_scratch()
        self.instrs.append({"load": [("const", mul_4097_const, 4097), ("const", mul_33_const, 33)]})
        self.instrs.append({"load": [("const", mul_2_const, 2), ("const", vlen_const, VLEN)]})
        self.const_map[4097] = mul_4097_const
        self.const_map[33] = mul_33_const
        self.const_map[2] = mul_2_const
        self.const_map[VLEN] = vlen_const
        self.const_map[9] = mul_9_const
        mul_4097_vec = self.alloc_scratch("mul_4097", VLEN)
        mul_33_vec = self.alloc_scratch("mul_33", VLEN)
        mul_9_vec = self.alloc_scratch("mul_9", VLEN)
        mul_2_vec = self.alloc_scratch("mul_2", VLEN)

        fused_D_const = self.alloc_scratch()
        fused_E_const = self.alloc_scratch()
        mul_16896_const = self.alloc_scratch()
        C2 = 0x165667B1
        C3 = 0xD3A2646C
        fused_D = (C2 + C3) % (2**32)
        fused_E = (C2 << 9) % (2**32)
        self.instrs.append({"load": [("const", fused_D_const, fused_D), ("const", fused_E_const, fused_E)]})
        self.instrs.append({"load": [("const", mul_16896_const, 16896), ("const", mul_9_const, 9)]})
        self.const_map[fused_D] = fused_D_const
        self.const_map[fused_E] = fused_E_const
        self.const_map[16896] = mul_16896_const

        fused_D_vec = self.alloc_scratch("fused_D", VLEN)
        fused_E_vec = self.alloc_scratch("fused_E", VLEN)
        mul_16896_vec = self.alloc_scratch("mul_16896", VLEN)

        i_const_0 = self.scratch_const(0)
        val1_vec_0, val3_vec_0, val1_const_0, val3_const_0 = hash_const_vecs[0]
        val1_vec_1, val3_vec_1, val1_const_1, val3_const_1 = hash_const_vecs[1]
        val1_vec_2, val3_vec_2, val1_const_2, val3_const_2 = hash_const_vecs[2]
        val1_vec_3, val3_vec_3, val1_const_3, val3_const_3 = hash_const_vecs[3]
        val1_vec_4, val3_vec_4, val1_const_4, val3_const_4 = hash_const_vecs[4]
        val1_vec_5, val3_vec_5, val1_const_5, val3_const_5 = hash_const_vecs[5]

        body.append({"valu": [("vbroadcast", one_vec, one_const),
                              ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]),
                              ("vbroadcast", forest_p_vec, self.scratch["forest_values_p"]),
                              ("vbroadcast", val1_vec_0, val1_const_0),
                              ("vbroadcast", mul_4097_vec, mul_4097_const),
                              ("vbroadcast", val1_vec_1, val1_const_1)],
                     "alu": [("+", addr_tmp, self.scratch["inp_indices_p"], i_const_0),
                             ("+", addr_tmp2, self.scratch["inp_values_p"], i_const_0)],
                     "load": [("load", broadcast_tmp, self.scratch["forest_values_p"])]})

        body.append({"load": [("vload", idx_chunks[0], addr_tmp),
                              ("vload", val_chunks[0], addr_tmp2)],
                     "alu": [("+", addr_tmp, addr_tmp, vlen_const),
                             ("+", addr_tmp2, addr_tmp2, vlen_const)],
                     "valu": [("vbroadcast", val3_vec_1, val3_const_1),
                              ("vbroadcast", val1_vec_2, val1_const_2),
                              ("vbroadcast", val1_vec_3, val1_const_3),
                              ("vbroadcast", val3_vec_3, val3_const_3),
                              ("vbroadcast", mul_33_vec, mul_33_const),
                              ("vbroadcast", val1_vec_4, val1_const_4)]})

        body.append({"load": [("vload", idx_chunks[1], addr_tmp),
                              ("vload", val_chunks[1], addr_tmp2)],
                     "alu": [("+", addr_tmp, addr_tmp, vlen_const),
                             ("+", addr_tmp2, addr_tmp2, vlen_const)],
                     "valu": [("vbroadcast", val1_vec_5, val1_const_5),
                              ("vbroadcast", val3_vec_5, val3_const_5),
                              ("vbroadcast", mul_9_vec, mul_9_const),
                              ("vbroadcast", mul_2_vec, mul_2_const),
                              ("vbroadcast", fused_D_vec, fused_D_const),
                              ("vbroadcast", fused_E_vec, fused_E_const)]})

        first_6_broadcasts = [("vbroadcast", node_val_vecs[cc], broadcast_tmp) for cc in range(5)] + [("vbroadcast", mul_16896_vec, mul_16896_const)]
        remaining_broadcasts = [("vbroadcast", node_val_vecs[cc], broadcast_tmp) for cc in range(5, n_chunks)]
        broadcast_idx = 0
        for c in range(2, n_chunks):
            instr = {"load": [("vload", idx_chunks[c], addr_tmp),
                              ("vload", val_chunks[c], addr_tmp2)],
                     "alu": [("+", addr_tmp, addr_tmp, vlen_const),
                             ("+", addr_tmp2, addr_tmp2, vlen_const)]}
            if c == n_chunks - 1:
                instr["valu"] = first_6_broadcasts
            elif broadcast_idx < len(remaining_broadcasts):
                take = min(6, len(remaining_broadcasts) - broadcast_idx)
                instr["valu"] = remaining_broadcasts[broadcast_idx:broadcast_idx + take]
                broadcast_idx += take
            body.append(instr)

        def get_stage_ops(c, stage, hash_t1s, hash_t2s, bounds_masks, n_temps, skip_addr=False):
            t = c % n_temps
            if stage == -1:
                return [("^", val_chunks[c], val_chunks[c], node_val_vecs[c])]
            elif stage == 0:
                return [("multiply_add", val_chunks[c], val_chunks[c], mul_4097_vec, val1_vec_0),
                        ("multiply_add", idx_chunks[c], idx_chunks[c], mul_2_vec, one_vec)]
            elif stage == 1:
                op1, _, _, op3, _ = HASH_STAGES[1]
                return [(op1, hash_t1s[t], val_chunks[c], val1_vec_1),
                        (op3, hash_t2s[t], val_chunks[c], val3_vec_1)]
            elif stage == 2:
                _, _, op2, _, _ = HASH_STAGES[1]
                return [(op2, val_chunks[c], hash_t1s[t], hash_t2s[t])]
            elif stage == 3:
                return [("multiply_add", hash_t1s[t], val_chunks[c], mul_33_vec, fused_D_vec),
                        ("multiply_add", hash_t2s[t], val_chunks[c], mul_16896_vec, fused_E_vec)]
            elif stage == 4:
                return [("^", val_chunks[c], hash_t1s[t], hash_t2s[t])]
            elif stage == 5:
                return [("multiply_add", val_chunks[c], val_chunks[c], mul_9_vec, val1_vec_4)]
            elif stage == 6:
                op1, _, _, op3, _ = HASH_STAGES[5]
                return [(op1, hash_t1s[t], val_chunks[c], val1_vec_5),
                        (op3, hash_t2s[t], val_chunks[c], val3_vec_5)]
            elif stage == 7:
                _, _, op2, _, _ = HASH_STAGES[5]
                return [(op2, val_chunks[c], hash_t1s[t], hash_t2s[t])]
            elif stage == 8:
                return [("&", hash_t2s[t], val_chunks[c], one_vec)]
            elif stage == 9:
                return [("+", idx_chunks[c], idx_chunks[c], hash_t2s[t]),
                        ("<", bounds_masks[t], idx_chunks[c], n_nodes_vec)]
            elif stage == 10:
                if skip_addr:
                    return [("*", idx_chunks[c], idx_chunks[c], bounds_masks[t])]
                return [("*", idx_chunks[c], idx_chunks[c], bounds_masks[t]),
                        ("multiply_add", addr_vecs[c], idx_chunks[c], bounds_masks[t], forest_p_vec)]
            return []

        gather_node_vals = [self.alloc_scratch(f"gather_nv_{i}", VLEN) for i in range(2)]
        gather_diff = self.alloc_scratch("gather_diff", VLEN)

        prefetched_round1 = False
        for round_num in range(rounds):
            is_last_round = (round_num == rounds - 1)

            use_broadcast = (round_num == 0)
            use_limited_gather_2 = (round_num == 1)

            chunk_stage = [-1] * n_chunks
            chunk_loaded = [False] * n_chunks

            if use_broadcast:
                for c in range(n_chunks):
                    chunk_loaded[c] = True

                round0_cycles = []
                while any(chunk_stage[c] < 11 for c in range(n_chunks)):
                    valu_batch = []
                    for cc in range(n_chunks):
                        if chunk_loaded[cc] and chunk_stage[cc] < 12 and len(valu_batch) < 6:
                            ops = get_stage_ops(cc, chunk_stage[cc], hash_t1s, hash_t2s, bounds_masks, n_temps, is_last_round)
                            if len(valu_batch) + len(ops) <= 6:
                                valu_batch.extend(ops)
                                chunk_stage[cc] += 1
                    if valu_batch:
                        round0_cycles.append({"valu": valu_batch})
                    else:
                        break

                if len(round0_cycles) >= 2:
                    round0_cycles[-2]["alu"] = [("+", addr_tmp, self.scratch["forest_values_p"], one_const),
                                                 ("+", addr_tmp2, self.scratch["forest_values_p"], mul_2_const)]
                    round0_cycles[-1]["load"] = [("load", gather_node_vals[0], addr_tmp),
                                                  ("load", gather_node_vals[0] + 1, addr_tmp2)]
                    body.extend(round0_cycles)
                    prefetched_round1 = True
                else:
                    body.extend(round0_cycles)
                    prefetched_round1 = False
            elif use_limited_gather_2:
                if not prefetched_round1:
                    body.append({"alu": [("+", addr_tmp, self.scratch["forest_values_p"], one_const),
                                         ("+", addr_tmp2, self.scratch["forest_values_p"], mul_2_const)]})
                    body.append({"load": [("load", gather_node_vals[0], addr_tmp),
                                          ("load", gather_node_vals[0] + 1, addr_tmp2)]})
                first_5_masks = [("&", hash_t1s[c % n_temps], idx_chunks[c], one_vec) for c in range(min(5, n_chunks))]
                body.append({"valu": [("vbroadcast", gather_node_vals[0], gather_node_vals[0]),
                                      ("vbroadcast", gather_node_vals[1], gather_node_vals[0] + 1)] + first_5_masks[:4]})
                remaining_first_masks = [("&", hash_t1s[c % n_temps], idx_chunks[c], one_vec) for c in range(4, min(6, n_chunks))]
                body.append({"valu": [("-", gather_diff, gather_node_vals[0], gather_node_vals[1])] + remaining_first_masks})

                for c in range(n_chunks):
                    chunk_loaded[c] = True

                first_6_selects = [("multiply_add", node_val_vecs[c], gather_diff, hash_t1s[c % n_temps], gather_node_vals[1]) for c in range(min(6, n_chunks))]
                body.append({"valu": first_6_selects})

                for batch_start in range(6, n_chunks, n_temps):
                    batch_end = min(batch_start + n_temps, n_chunks)
                    mask_ops = []
                    for c in range(batch_start, batch_end):
                        t = c % n_temps
                        mask_ops.append(("&", hash_t1s[t], idx_chunks[c], one_vec))
                    body.append({"valu": mask_ops})

                    select_ops = []
                    for c in range(batch_start, batch_end):
                        t = c % n_temps
                        select_ops.append(("multiply_add", node_val_vecs[c], gather_diff, hash_t1s[t], gather_node_vals[1]))
                    body.append({"valu": select_ops})

                while any(chunk_stage[cc] < 11 for cc in range(n_chunks)):
                    valu_batch = []
                    for cc in range(n_chunks):
                        if chunk_loaded[cc] and chunk_stage[cc] < 12 and len(valu_batch) < 6:
                            ops = get_stage_ops(cc, chunk_stage[cc], hash_t1s, hash_t2s, bounds_masks, n_temps, is_last_round)
                            if len(valu_batch) + len(ops) <= 6:
                                valu_batch.extend(ops)
                                chunk_stage[cc] += 1
                    if valu_batch:
                        body.append({"valu": valu_batch})
                    else:
                        break
            else:
                for c in range(n_chunks):
                    addr_vec = addr_vecs[c]
                    node_val_vec = node_val_vecs[c]

                    load_cycles = [
                        [("load", node_val_vec, addr_vec), ("load", node_val_vec + 1, addr_vec + 1)],
                        [("load", node_val_vec + 2, addr_vec + 2), ("load", node_val_vec + 3, addr_vec + 3)],
                        [("load", node_val_vec + 4, addr_vec + 4), ("load", node_val_vec + 5, addr_vec + 5)],
                        [("load", node_val_vec + 6, addr_vec + 6), ("load", node_val_vec + 7, addr_vec + 7)],
                    ]

                    for load_idx, load_ops in enumerate(load_cycles):
                        valu_batch = []

                        for earlier_c in range(c):
                            if chunk_loaded[earlier_c] and chunk_stage[earlier_c] < 12 and len(valu_batch) < 6:
                                ops = get_stage_ops(earlier_c, chunk_stage[earlier_c], hash_t1s, hash_t2s, bounds_masks, n_temps, is_last_round)
                                if len(valu_batch) + len(ops) <= 6:
                                    valu_batch.extend(ops)
                                    chunk_stage[earlier_c] += 1

                        body.append({"load": load_ops, "valu": valu_batch} if valu_batch else {"load": load_ops})

                    chunk_loaded[c] = True

                cleanup_cycles = []
                while any(chunk_stage[c] < 11 for c in range(n_chunks)):
                    valu_batch = []
                    for cc in range(n_chunks):
                        if chunk_loaded[cc] and chunk_stage[cc] < 12 and len(valu_batch) < 6:
                            ops = get_stage_ops(cc, chunk_stage[cc], hash_t1s, hash_t2s, bounds_masks, n_temps, is_last_round)
                            if len(valu_batch) + len(ops) <= 6:
                                valu_batch.extend(ops)
                                chunk_stage[cc] += 1
                    if valu_batch:
                        cleanup_cycles.append({"valu": valu_batch})
                    else:
                        break
                if is_last_round and cleanup_cycles:
                    cleanup_cycles[-1]["alu"] = [("+", addr_tmp, self.scratch["inp_indices_p"], i_const_0),
                                                  ("+", addr_tmp2, self.scratch["inp_values_p"], i_const_0)]
                body.extend(cleanup_cycles)

            if is_last_round and not (round_num > 0 and cleanup_cycles):
                body.append({"alu": [("+", addr_tmp, self.scratch["inp_indices_p"], i_const_0),
                                     ("+", addr_tmp2, self.scratch["inp_values_p"], i_const_0)]})

        body.append({"store": [("vstore", addr_tmp, idx_chunks[0]),
                               ("vstore", addr_tmp2, val_chunks[0])],
                     "alu": [("+", addr_tmp, addr_tmp, vlen_const),
                             ("+", addr_tmp2, addr_tmp2, vlen_const)]})

        for c in range(1, n_chunks - 1):
            body.append({"store": [("vstore", addr_tmp, idx_chunks[c]),
                                   ("vstore", addr_tmp2, val_chunks[c])],
                         "alu": [("+", addr_tmp, addr_tmp, vlen_const),
                                 ("+", addr_tmp2, addr_tmp2, vlen_const)]})
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
