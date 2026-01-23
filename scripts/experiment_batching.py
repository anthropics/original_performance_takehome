#!/usr/bin/env python3
"""
Experiment: Compare different batch sizes (2, 3, 4 batches per triple).
"""

import sys

sys.path.insert(0, ".")

from problem import Tree, Input, build_mem_image, Machine, N_CORES, VLEN, HASH_STAGES
from perf_takehome import KernelBuilder


def build_kernel_with_batch_size(
    batch_count, forest_height, n_nodes, batch_size, rounds
):
    """Build kernel with specified batch count per triple."""
    kb = KernelBuilder()

    tmp1 = kb.alloc_scratch("tmp1")
    tmp2 = kb.alloc_scratch("tmp2")
    tmp3 = kb.alloc_scratch("tmp3")

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
        kb.alloc_scratch(v, 1)
    for i, v in enumerate(init_vars):
        kb.add("load", ("const", tmp1, i))
        kb.add("load", ("load", kb.scratch[v], tmp1))

    zero_const = kb.scratch_const(0)
    one_const = kb.scratch_const(1)
    two_const = kb.scratch_const(2)

    hash_consts = []
    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        c1 = kb.alloc_scratch(f"hc1_{hi}", VLEN)
        c3 = kb.alloc_scratch(f"hc3_{hi}", VLEN)
        kb.add("load", ("const", tmp1, val1))
        kb.emit({"valu": [("vbroadcast", c1, tmp1)]})
        kb.add("load", ("const", tmp1, val3))
        kb.emit({"valu": [("vbroadcast", c3, tmp1)]})
        hash_consts.append((c1, c3))

    v_zero = kb.alloc_scratch("v_zero", VLEN)
    v_one = kb.alloc_scratch("v_one", VLEN)
    v_two = kb.alloc_scratch("v_two", VLEN)
    v_n_nodes = kb.alloc_scratch("v_n_nodes", VLEN)
    v_forest_p = kb.alloc_scratch("v_forest_p", VLEN)

    kb.emit({"valu": [("vbroadcast", v_zero, zero_const)]})
    kb.emit({"valu": [("vbroadcast", v_one, one_const)]})
    kb.emit({"valu": [("vbroadcast", v_two, two_const)]})
    kb.emit({"valu": [("vbroadcast", v_n_nodes, kb.scratch["n_nodes"])]})
    kb.emit({"valu": [("vbroadcast", v_forest_p, kb.scratch["forest_values_p"])]})

    n_vectors = batch_size // VLEN
    s_idx = kb.alloc_scratch("s_idx", n_vectors * VLEN)
    s_val = kb.alloc_scratch("s_val", n_vectors * VLEN)

    B = batch_count
    v_val = [kb.alloc_scratch(f"v_val_{i}", VLEN) for i in range(B)]
    v_tmp1 = [kb.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(B)]
    v_tmp2 = [kb.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(B)]
    v_idx = [kb.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(B)]
    v_node = [kb.alloc_scratch(f"v_node_{i}", VLEN) for i in range(B)]
    v_addr = [kb.alloc_scratch(f"v_addr_{i}", VLEN) for i in range(B)]
    v_node_next = [kb.alloc_scratch(f"v_node_next_{i}", VLEN) for i in range(B)]
    v_addr_next = [kb.alloc_scratch(f"v_addr_next_{i}", VLEN) for i in range(B)]

    v_broadcast_node = kb.alloc_scratch("v_broadcast_node", VLEN)
    addr_tmp = kb.alloc_scratch("addr_tmp")

    kb.add("flow", ("pause",))

    for vi in range(n_vectors):
        off = vi * VLEN
        kb.emit({"load": [("const", addr_tmp, off)]})
        kb.emit({"alu": [("+", addr_tmp, kb.scratch["inp_indices_p"], addr_tmp)]})
        kb.emit({"load": [("vload", s_idx + off, addr_tmp)]})

    for vi in range(n_vectors):
        off = vi * VLEN
        kb.emit({"load": [("const", addr_tmp, off)]})
        kb.emit({"alu": [("+", addr_tmp, kb.scratch["inp_values_p"], addr_tmp)]})
        kb.emit({"load": [("vload", s_val + off, addr_tmp)]})

    broadcast_rounds = {0, 11}

    for rnd in range(rounds):
        is_broadcast = rnd in broadcast_rounds

        if is_broadcast:
            kb.emit({"load": [("load", tmp1, kb.scratch["forest_values_p"])]})
            kb.emit({"valu": [("vbroadcast", v_broadcast_node, tmp1)]})

            n_groups = n_vectors // B
            remainder = n_vectors % B

            for group in range(n_groups):
                offs = [(group * B + i) * VLEN for i in range(B)]
                kb.emit(
                    {
                        "valu": [
                            ("^", v_val[i], s_val + offs[i], v_broadcast_node)
                            for i in range(B)
                        ]
                    }
                )

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1, c3 = hash_consts[hi]
                    ops = [(op1, v_tmp1[i], v_val[i], c1) for i in range(B)]
                    ops += [(op3, v_tmp2[i], v_val[i], c3) for i in range(B)]
                    kb.emit({"valu": ops[:6]})
                    if len(ops) > 6:
                        kb.emit({"valu": ops[6:]})
                    kb.emit(
                        {
                            "valu": [
                                (op2, v_val[i], v_tmp1[i], v_tmp2[i]) for i in range(B)
                            ]
                        }
                    )

                kb.emit({"valu": [("&", v_tmp1[i], v_val[i], v_one) for i in range(B)]})
                kb.emit(
                    {"valu": [("+", v_tmp2[i], v_one, v_tmp1[i]) for i in range(B)]}
                )
                kb.emit(
                    {
                        "valu": [
                            (
                                "multiply_add",
                                v_idx[i],
                                s_idx + offs[i],
                                v_two,
                                v_tmp2[i],
                            )
                            for i in range(B)
                        ]
                    }
                )
                kb.emit(
                    {"valu": [("<", v_tmp1[i], v_idx[i], v_n_nodes) for i in range(B)]}
                )
                kb.emit(
                    {"valu": [("*", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(B)]}
                )

                wb_ops = [("+", s_idx + offs[i], v_idx[i], v_zero) for i in range(B)]
                wb_ops += [("+", s_val + offs[i], v_val[i], v_zero) for i in range(B)]
                kb.emit({"valu": wb_ops[:6]})
                if len(wb_ops) > 6:
                    kb.emit({"valu": wb_ops[6:]})

            for i in range(remainder):
                off = (n_groups * B + i) * VLEN
                kb.emit({"valu": [("^", v_val[0], s_val + off, v_broadcast_node)]})
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1, c3 = hash_consts[hi]
                    kb.emit(
                        {
                            "valu": [
                                (op1, v_tmp1[0], v_val[0], c1),
                                (op3, v_tmp2[0], v_val[0], c3),
                            ]
                        }
                    )
                    kb.emit({"valu": [(op2, v_val[0], v_tmp1[0], v_tmp2[0])]})
                kb.emit({"valu": [("&", v_tmp1[0], v_val[0], v_one)]})
                kb.emit({"valu": [("+", v_tmp2[0], v_one, v_tmp1[0])]})
                kb.emit(
                    {
                        "valu": [
                            ("multiply_add", v_idx[0], s_idx + off, v_two, v_tmp2[0])
                        ]
                    }
                )
                kb.emit({"valu": [("<", v_tmp1[0], v_idx[0], v_n_nodes)]})
                kb.emit({"valu": [("*", v_idx[0], v_idx[0], v_tmp1[0])]})
                kb.emit(
                    {
                        "valu": [
                            ("+", s_idx + off, v_idx[0], v_zero),
                            ("+", s_val + off, v_val[0], v_zero),
                        ]
                    }
                )
        else:
            n_groups = n_vectors // B
            remainder = n_vectors % B

            for group in range(n_groups):
                offs = [(group * B + i) * VLEN for i in range(B)]
                has_next = group < n_groups - 1 or remainder > 0

                if group == 0:
                    kb.emit(
                        {
                            "valu": [
                                ("+", v_addr[i], v_forest_p, s_idx + offs[i])
                                for i in range(B)
                            ]
                        }
                    )
                    for elem in range(VLEN):
                        loads = [
                            ("load_offset", v_node[i], v_addr[i], elem)
                            for i in range(min(2, B))
                        ]
                        kb.emit({"load": loads})
                    if B > 2:
                        for elem in range(VLEN):
                            loads = [
                                ("load_offset", v_node[i], v_addr[i], elem)
                                for i in range(2, B)
                            ]
                            kb.emit({"load": loads})

                kb.emit(
                    {
                        "valu": [
                            ("^", v_val[i], s_val + offs[i], v_node[i])
                            for i in range(B)
                        ]
                    }
                )

                if has_next:
                    if group < n_groups - 1:
                        next_offs = [((group + 1) * B + i) * VLEN for i in range(B)]
                        next_count = B
                    else:
                        next_offs = [
                            (n_groups * B + i) * VLEN for i in range(remainder)
                        ]
                        next_count = remainder
                    kb.emit(
                        {
                            "valu": [
                                ("+", v_addr_next[i], v_forest_p, s_idx + next_offs[i])
                                for i in range(next_count)
                            ]
                        }
                    )

                prefetch_idx = 0
                next_count = (
                    B
                    if (has_next and group < n_groups - 1)
                    else (remainder if has_next else 0)
                )

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1, c3 = hash_consts[hi]
                    loads = []
                    for _ in range(2):
                        if has_next and prefetch_idx < VLEN * next_count:
                            bi, ei = prefetch_idx // VLEN, prefetch_idx % VLEN
                            if bi < next_count:
                                loads.append(
                                    (
                                        "load_offset",
                                        v_node_next[bi],
                                        v_addr_next[bi],
                                        ei,
                                    )
                                )
                                prefetch_idx += 1

                    ops = [(op1, v_tmp1[i], v_val[i], c1) for i in range(B)]
                    ops += [(op3, v_tmp2[i], v_val[i], c3) for i in range(B)]
                    kb.emit({"valu": ops[:6], **({"load": loads} if loads else {})})
                    if len(ops) > 6:
                        kb.emit({"valu": ops[6:]})

                    loads = []
                    for _ in range(2):
                        if has_next and prefetch_idx < VLEN * next_count:
                            bi, ei = prefetch_idx // VLEN, prefetch_idx % VLEN
                            if bi < next_count:
                                loads.append(
                                    (
                                        "load_offset",
                                        v_node_next[bi],
                                        v_addr_next[bi],
                                        ei,
                                    )
                                )
                                prefetch_idx += 1
                    kb.emit(
                        {
                            "valu": [
                                (op2, v_val[i], v_tmp1[i], v_tmp2[i]) for i in range(B)
                            ],
                            **({"load": loads} if loads else {}),
                        }
                    )

                while has_next and prefetch_idx < VLEN * next_count:
                    loads = []
                    for _ in range(2):
                        if prefetch_idx < VLEN * next_count:
                            bi, ei = prefetch_idx // VLEN, prefetch_idx % VLEN
                            if bi < next_count:
                                loads.append(
                                    (
                                        "load_offset",
                                        v_node_next[bi],
                                        v_addr_next[bi],
                                        ei,
                                    )
                                )
                                prefetch_idx += 1
                    if loads:
                        kb.emit({"load": loads})

                kb.emit({"valu": [("&", v_tmp1[i], v_val[i], v_one) for i in range(B)]})
                kb.emit(
                    {"valu": [("+", v_tmp2[i], v_one, v_tmp1[i]) for i in range(B)]}
                )
                kb.emit(
                    {
                        "valu": [
                            (
                                "multiply_add",
                                v_idx[i],
                                s_idx + offs[i],
                                v_two,
                                v_tmp2[i],
                            )
                            for i in range(B)
                        ]
                    }
                )
                kb.emit(
                    {"valu": [("<", v_tmp1[i], v_idx[i], v_n_nodes) for i in range(B)]}
                )
                kb.emit(
                    {"valu": [("*", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(B)]}
                )

                wb_ops = [("+", s_idx + offs[i], v_idx[i], v_zero) for i in range(B)]
                wb_ops += [("+", s_val + offs[i], v_val[i], v_zero) for i in range(B)]
                kb.emit({"valu": wb_ops[:6]})
                if len(wb_ops) > 6:
                    kb.emit({"valu": wb_ops[6:]})

                if has_next:
                    v_node, v_node_next = v_node_next, v_node
                    v_addr, v_addr_next = v_addr_next, v_addr

            if remainder > 0:
                offs = [(n_groups * B + i) * VLEN for i in range(remainder)]
                if n_groups == 0:
                    for i in range(remainder):
                        kb.emit(
                            {"valu": [("+", v_addr[i], v_forest_p, s_idx + offs[i])]}
                        )
                    for b in range(remainder):
                        for elem in range(VLEN):
                            kb.emit(
                                {"load": [("load_offset", v_node[b], v_addr[b], elem)]}
                            )

                for i in range(remainder):
                    kb.emit({"valu": [("^", v_val[i], s_val + offs[i], v_node[i])]})

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1, c3 = hash_consts[hi]
                    ops1 = [(op1, v_tmp1[i], v_val[i], c1) for i in range(remainder)]
                    ops1 += [(op3, v_tmp2[i], v_val[i], c3) for i in range(remainder)]
                    kb.emit({"valu": ops1})
                    kb.emit(
                        {
                            "valu": [
                                (op2, v_val[i], v_tmp1[i], v_tmp2[i])
                                for i in range(remainder)
                            ]
                        }
                    )

                for i in range(remainder):
                    kb.emit({"valu": [("&", v_tmp1[i], v_val[i], v_one)]})
                    kb.emit({"valu": [("+", v_tmp2[i], v_one, v_tmp1[i])]})
                    kb.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_idx[i],
                                    s_idx + offs[i],
                                    v_two,
                                    v_tmp2[i],
                                )
                            ]
                        }
                    )
                    kb.emit({"valu": [("<", v_tmp1[i], v_idx[i], v_n_nodes)]})
                    kb.emit({"valu": [("*", v_idx[i], v_idx[i], v_tmp1[i])]})
                    kb.emit(
                        {
                            "valu": [
                                ("+", s_idx + offs[i], v_idx[i], v_zero),
                                ("+", s_val + offs[i], v_val[i], v_zero),
                            ]
                        }
                    )

    for vi in range(n_vectors):
        off = vi * VLEN
        kb.emit({"load": [("const", addr_tmp, off)]})
        kb.emit({"alu": [("+", addr_tmp, kb.scratch["inp_indices_p"], addr_tmp)]})
        kb.emit({"store": [("vstore", addr_tmp, s_idx + off)]})

    for vi in range(n_vectors):
        off = vi * VLEN
        kb.emit({"load": [("const", addr_tmp, off)]})
        kb.emit({"alu": [("+", addr_tmp, kb.scratch["inp_values_p"], addr_tmp)]})
        kb.emit({"store": [("vstore", addr_tmp, s_val + off)]})

    kb.emit({"flow": [("pause",)]})

    return kb


def main():
    import random

    print("=" * 60)
    print("BATCH SIZE EXPERIMENT")
    print("=" * 60)

    for batch_count in [2, 3, 4]:
        print(f"\n--- Batch count: {batch_count} ---")

        try:
            kb = build_kernel_with_batch_size(batch_count, 10, 1023, 256, 16)
            print(f"Bundles: {len(kb.instrs)}")
            print(f"Scratch: {kb.scratch_ptr} / 1536")

            random.seed(123)
            forest = Tree.generate(10)
            inp = Input.generate(forest, 256, 16)
            mem = build_mem_image(forest, inp)

            machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
            machine.run()

            print(f"Cycles: {machine.cycle}")
            print(f"Speedup: {147734 / machine.cycle:.1f}x")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
