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

        # Analyze which hash stages can be fused:
        # Stages where op1='+', op2='+', op3='<<' can be fused:
        #   a = (a + c1) + (a << k) = a * (1 + 2^k) + c1 = multiply_add(a, factor, c1)
        # Stages 0, 2, 4 are fusable; stages 1, 3, 5 are not (XOR patterns)
        hash_consts = []
        fusable_stages = set()
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # Fusable: multiply_add(a, 1+2^val3, val1)
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
                hash_consts.append((c_factor, c_const, True))  # True = fusable
            else:
                # Not fusable: keep original 3-op pattern
                c1 = self.alloc_scratch(f"hc1_{hi}", VLEN)
                c3 = self.alloc_scratch(f"hc3_{hi}", VLEN)
                self.emit({"load": [("const", tmp1, val1), ("const", tmp2, val3)]})
                self.emit(
                    {"valu": [("vbroadcast", c1, tmp1), ("vbroadcast", c3, tmp2)]}
                )
                hash_consts.append((c1, c3, False))  # False = not fusable

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        self.emit(
            {
                "valu": [
                    ("vbroadcast", v_zero, zero_const),
                    ("vbroadcast", v_one, one_const),
                    ("vbroadcast", v_two, two_const),
                    ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
                    ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
                ]
            }
        )

        n_vectors = batch_size // VLEN
        s_idx = self.alloc_scratch("s_idx", n_vectors * VLEN)
        s_val = self.alloc_scratch("s_val", n_vectors * VLEN)

        BATCH = 6
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(BATCH)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(BATCH)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(BATCH)]
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(BATCH)]
        v_node = [self.alloc_scratch(f"v_node_{i}", VLEN) for i in range(BATCH)]
        v_addr = [self.alloc_scratch(f"v_addr_{i}", VLEN) for i in range(BATCH)]
        v_node_next = [
            self.alloc_scratch(f"v_node_next_{i}", VLEN) for i in range(BATCH)
        ]
        v_addr_next = [
            self.alloc_scratch(f"v_addr_next_{i}", VLEN) for i in range(BATCH)
        ]

        v_broadcast_node = self.alloc_scratch("v_broadcast_node", VLEN)
        addr_tmp = self.alloc_scratch("addr_tmp")

        v_tree_1 = self.alloc_scratch("v_tree_1", VLEN)
        v_tree_2 = self.alloc_scratch("v_tree_2", VLEN)
        v_tree_diff = self.alloc_scratch("v_tree_diff", VLEN)

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
        select_rounds = {1, 12}

        for rnd in range(rounds):
            is_broadcast_round = rnd in broadcast_rounds
            is_select_round = rnd in select_rounds

            if is_broadcast_round:
                self.emit({"load": [("load", tmp1, self.scratch["forest_values_p"])]})
                self.emit({"valu": [("vbroadcast", v_broadcast_node, tmp1)]})

                n_batches_bc = n_vectors // BATCH
                remainder_bc = n_vectors % BATCH

                for batch in range(n_batches_bc):
                    offs = [(batch * BATCH + i) * VLEN for i in range(BATCH)]
                    self.emit(
                        {
                            "valu": [
                                ("^", v_val[i], s_val + offs[i], v_broadcast_node)
                                for i in range(BATCH)
                            ]
                        }
                    )
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        hc = hash_consts[hi]
                        if hc[2]:  # fusable
                            c_factor, c_const = hc[0], hc[1]
                            self.emit(
                                {
                                    "valu": [
                                        (
                                            "multiply_add",
                                            v_val[i],
                                            v_val[i],
                                            c_factor,
                                            c_const,
                                        )
                                        for i in range(BATCH)
                                    ]
                                }
                            )
                        else:  # non-fusable (12 ops = 2 bundles, then 6 ops = 1 bundle)
                            c1, c3 = hc[0], hc[1]
                            self.emit(
                                {
                                    "valu": [
                                        (op1, v_tmp1[i], v_val[i], c1)
                                        for i in range(BATCH)
                                    ]
                                }
                            )
                            self.emit(
                                {
                                    "valu": [
                                        (op3, v_tmp2[i], v_val[i], c3)
                                        for i in range(BATCH)
                                    ]
                                }
                            )
                            self.emit(
                                {
                                    "valu": [
                                        (op2, v_val[i], v_tmp1[i], v_tmp2[i])
                                        for i in range(BATCH)
                                    ]
                                }
                            )
                    self.emit(
                        {
                            "valu": [
                                ("&", v_tmp1[i], v_val[i], v_one) for i in range(BATCH)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", v_tmp2[i], v_one, v_tmp1[i]) for i in range(BATCH)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_idx[i],
                                    s_idx + offs[i],
                                    v_two,
                                    v_tmp2[i],
                                )
                                for i in range(BATCH)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("<", v_tmp1[i], v_idx[i], v_n_nodes)
                                for i in range(BATCH)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("*", v_idx[i], v_idx[i], v_tmp1[i])
                                for i in range(BATCH)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", s_idx + offs[i], v_idx[i], v_zero)
                                for i in range(BATCH)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", s_val + offs[i], v_val[i], v_zero)
                                for i in range(BATCH)
                            ]
                        }
                    )

                if remainder_bc > 0:
                    offs_bc = [
                        (n_batches_bc * BATCH + i) * VLEN for i in range(remainder_bc)
                    ]
                    self.emit(
                        {
                            "valu": [
                                ("^", v_val[i], s_val + offs_bc[i], v_broadcast_node)
                                for i in range(remainder_bc)
                            ]
                        }
                    )
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        hc = hash_consts[hi]
                        if hc[2]:  # fusable
                            c_factor, c_const = hc[0], hc[1]
                            self.emit(
                                {
                                    "valu": [
                                        (
                                            "multiply_add",
                                            v_val[i],
                                            v_val[i],
                                            c_factor,
                                            c_const,
                                        )
                                        for i in range(remainder_bc)
                                    ]
                                }
                            )
                        else:
                            c1, c3 = hc[0], hc[1]
                            ops1 = [
                                (op1, v_tmp1[i], v_val[i], c1)
                                for i in range(remainder_bc)
                            ]
                            ops1 += [
                                (op3, v_tmp2[i], v_val[i], c3)
                                for i in range(remainder_bc)
                            ]
                            self.emit({"valu": ops1})
                            self.emit(
                                {
                                    "valu": [
                                        (op2, v_val[i], v_tmp1[i], v_tmp2[i])
                                        for i in range(remainder_bc)
                                    ]
                                }
                            )
                    self.emit(
                        {
                            "valu": [
                                ("&", v_tmp1[i], v_val[i], v_one)
                                for i in range(remainder_bc)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", v_tmp2[i], v_one, v_tmp1[i])
                                for i in range(remainder_bc)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_idx[i],
                                    s_idx + offs_bc[i],
                                    v_two,
                                    v_tmp2[i],
                                )
                                for i in range(remainder_bc)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("<", v_tmp1[i], v_idx[i], v_n_nodes)
                                for i in range(remainder_bc)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("*", v_idx[i], v_idx[i], v_tmp1[i])
                                for i in range(remainder_bc)
                            ]
                        }
                    )
                    wb_bc = [
                        ("+", s_idx + offs_bc[i], v_idx[i], v_zero)
                        for i in range(remainder_bc)
                    ]
                    wb_bc += [
                        ("+", s_val + offs_bc[i], v_val[i], v_zero)
                        for i in range(remainder_bc)
                    ]
                    self.emit({"valu": wb_bc})
            elif is_select_round:
                n_triples_sel = n_vectors // 3
                remainder_sel = n_vectors % 3

                for triple in range(n_triples_sel):
                    offs = [(triple * 3 + i) * VLEN for i in range(3)]
                    self.emit(
                        {
                            "valu": [
                                ("-", v_tmp1[0], s_idx + offs[0], v_one),
                                ("-", v_tmp1[1], s_idx + offs[1], v_one),
                                ("-", v_tmp1[2], s_idx + offs[2], v_one),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_node[0],
                                    v_tmp1[0],
                                    v_tree_diff,
                                    v_tree_1,
                                ),
                                (
                                    "multiply_add",
                                    v_node[1],
                                    v_tmp1[1],
                                    v_tree_diff,
                                    v_tree_1,
                                ),
                                (
                                    "multiply_add",
                                    v_node[2],
                                    v_tmp1[2],
                                    v_tree_diff,
                                    v_tree_1,
                                ),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("^", v_val[0], s_val + offs[0], v_node[0]),
                                ("^", v_val[1], s_val + offs[1], v_node[1]),
                                ("^", v_val[2], s_val + offs[2], v_node[2]),
                            ]
                        }
                    )
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        hc = hash_consts[hi]
                        if hc[2]:
                            c_factor, c_const = hc[0], hc[1]
                            self.emit(
                                {
                                    "valu": [
                                        (
                                            "multiply_add",
                                            v_val[0],
                                            v_val[0],
                                            c_factor,
                                            c_const,
                                        ),
                                        (
                                            "multiply_add",
                                            v_val[1],
                                            v_val[1],
                                            c_factor,
                                            c_const,
                                        ),
                                        (
                                            "multiply_add",
                                            v_val[2],
                                            v_val[2],
                                            c_factor,
                                            c_const,
                                        ),
                                    ]
                                }
                            )
                        else:
                            c1, c3 = hc[0], hc[1]
                            self.emit(
                                {
                                    "valu": [
                                        (op1, v_tmp1[0], v_val[0], c1),
                                        (op1, v_tmp1[1], v_val[1], c1),
                                        (op1, v_tmp1[2], v_val[2], c1),
                                        (op3, v_tmp2[0], v_val[0], c3),
                                        (op3, v_tmp2[1], v_val[1], c3),
                                        (op3, v_tmp2[2], v_val[2], c3),
                                    ]
                                }
                            )
                            self.emit(
                                {
                                    "valu": [
                                        (op2, v_val[0], v_tmp1[0], v_tmp2[0]),
                                        (op2, v_val[1], v_tmp1[1], v_tmp2[1]),
                                        (op2, v_val[2], v_tmp1[2], v_tmp2[2]),
                                    ]
                                }
                            )
                    self.emit(
                        {
                            "valu": [
                                ("&", v_tmp1[0], v_val[0], v_one),
                                ("&", v_tmp1[1], v_val[1], v_one),
                                ("&", v_tmp1[2], v_val[2], v_one),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", v_tmp2[0], v_one, v_tmp1[0]),
                                ("+", v_tmp2[1], v_one, v_tmp1[1]),
                                ("+", v_tmp2[2], v_one, v_tmp1[2]),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_idx[0],
                                    s_idx + offs[0],
                                    v_two,
                                    v_tmp2[0],
                                ),
                                (
                                    "multiply_add",
                                    v_idx[1],
                                    s_idx + offs[1],
                                    v_two,
                                    v_tmp2[1],
                                ),
                                (
                                    "multiply_add",
                                    v_idx[2],
                                    s_idx + offs[2],
                                    v_two,
                                    v_tmp2[2],
                                ),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("<", v_tmp1[0], v_idx[0], v_n_nodes),
                                ("<", v_tmp1[1], v_idx[1], v_n_nodes),
                                ("<", v_tmp1[2], v_idx[2], v_n_nodes),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("*", v_idx[0], v_idx[0], v_tmp1[0]),
                                ("*", v_idx[1], v_idx[1], v_tmp1[1]),
                                ("*", v_idx[2], v_idx[2], v_tmp1[2]),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", s_idx + offs[0], v_idx[0], v_zero),
                                ("+", s_idx + offs[1], v_idx[1], v_zero),
                                ("+", s_idx + offs[2], v_idx[2], v_zero),
                                ("+", s_val + offs[0], v_val[0], v_zero),
                                ("+", s_val + offs[1], v_val[1], v_zero),
                                ("+", s_val + offs[2], v_val[2], v_zero),
                            ]
                        }
                    )

                if remainder_sel > 0:
                    offs_sel = [
                        (n_triples_sel * 3 + i) * VLEN for i in range(remainder_sel)
                    ]
                    self.emit(
                        {
                            "valu": [
                                ("-", v_tmp1[i], s_idx + offs_sel[i], v_one)
                                for i in range(remainder_sel)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_node[i],
                                    v_tmp1[i],
                                    v_tree_diff,
                                    v_tree_1,
                                )
                                for i in range(remainder_sel)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("^", v_val[i], s_val + offs_sel[i], v_node[i])
                                for i in range(remainder_sel)
                            ]
                        }
                    )
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        hc = hash_consts[hi]
                        if hc[2]:
                            c_factor, c_const = hc[0], hc[1]
                            self.emit(
                                {
                                    "valu": [
                                        (
                                            "multiply_add",
                                            v_val[i],
                                            v_val[i],
                                            c_factor,
                                            c_const,
                                        )
                                        for i in range(remainder_sel)
                                    ]
                                }
                            )
                        else:
                            c1, c3 = hc[0], hc[1]
                            ops1 = [
                                (op1, v_tmp1[i], v_val[i], c1)
                                for i in range(remainder_sel)
                            ]
                            ops1 += [
                                (op3, v_tmp2[i], v_val[i], c3)
                                for i in range(remainder_sel)
                            ]
                            self.emit({"valu": ops1})
                            self.emit(
                                {
                                    "valu": [
                                        (op2, v_val[i], v_tmp1[i], v_tmp2[i])
                                        for i in range(remainder_sel)
                                    ]
                                }
                            )
                    self.emit(
                        {
                            "valu": [
                                ("&", v_tmp1[i], v_val[i], v_one)
                                for i in range(remainder_sel)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", v_tmp2[i], v_one, v_tmp1[i])
                                for i in range(remainder_sel)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_idx[i],
                                    s_idx + offs_sel[i],
                                    v_two,
                                    v_tmp2[i],
                                )
                                for i in range(remainder_sel)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("<", v_tmp1[i], v_idx[i], v_n_nodes)
                                for i in range(remainder_sel)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("*", v_idx[i], v_idx[i], v_tmp1[i])
                                for i in range(remainder_sel)
                            ]
                        }
                    )
                    wb_sel = [
                        ("+", s_idx + offs_sel[i], v_idx[i], v_zero)
                        for i in range(remainder_sel)
                    ]
                    wb_sel += [
                        ("+", s_val + offs_sel[i], v_val[i], v_zero)
                        for i in range(remainder_sel)
                    ]
                    self.emit({"valu": wb_sel})
            else:
                n_triples = n_vectors // 3
                remainder = n_vectors % 3

                for triple in range(n_triples):
                    offs = [(triple * 3 + i) * VLEN for i in range(3)]
                    has_next = triple < n_triples - 1 or remainder > 0
                    next_offs = []
                    if has_next:
                        if triple < n_triples - 1:
                            next_offs = [
                                ((triple + 1) * 3 + i) * VLEN for i in range(3)
                            ]
                        else:
                            next_offs = [
                                (n_triples * 3 + i) * VLEN for i in range(remainder)
                            ]

                    if triple == 0:
                        self.emit(
                            {
                                "valu": [
                                    ("+", v_addr[0], v_forest_p, s_idx + offs[0]),
                                    ("+", v_addr[1], v_forest_p, s_idx + offs[1]),
                                    ("+", v_addr[2], v_forest_p, s_idx + offs[2]),
                                ]
                            }
                        )
                        for i in range(VLEN):
                            self.emit(
                                {
                                    "load": [
                                        ("load_offset", v_node[0], v_addr[0], i),
                                        ("load_offset", v_node[1], v_addr[1], i),
                                    ]
                                }
                            )
                        for i in range(0, VLEN, 2):
                            if i + 1 < VLEN:
                                self.emit(
                                    {
                                        "load": [
                                            ("load_offset", v_node[2], v_addr[2], i),
                                            (
                                                "load_offset",
                                                v_node[2],
                                                v_addr[2],
                                                i + 1,
                                            ),
                                        ]
                                    }
                                )
                            else:
                                self.emit(
                                    {"load": [("load_offset", v_node[2], v_addr[2], i)]}
                                )

                    if has_next:
                        next_count = 3 if triple < n_triples - 1 else remainder
                        xor_ops = [
                            ("^", v_val[0], s_val + offs[0], v_node[0]),
                            ("^", v_val[1], s_val + offs[1], v_node[1]),
                            ("^", v_val[2], s_val + offs[2], v_node[2]),
                        ]
                        addr_ops = [
                            ("+", v_addr_next[i], v_forest_p, s_idx + next_offs[i])
                            for i in range(next_count)
                        ]
                        self.emit({"valu": xor_ops + addr_ops})
                    else:
                        self.emit(
                            {
                                "valu": [
                                    ("^", v_val[0], s_val + offs[0], v_node[0]),
                                    ("^", v_val[1], s_val + offs[1], v_node[1]),
                                    ("^", v_val[2], s_val + offs[2], v_node[2]),
                                ]
                            }
                        )

                    prefetch_idx = 0
                    next_count = (
                        3
                        if (has_next and triple < n_triples - 1)
                        else (remainder if has_next else 0)
                    )

                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        hc = hash_consts[hi]
                        if hc[2]:  # fusable stage
                            c_factor, c_const = hc[0], hc[1]
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
                            self.emit(
                                {
                                    "valu": [
                                        (
                                            "multiply_add",
                                            v_val[0],
                                            v_val[0],
                                            c_factor,
                                            c_const,
                                        ),
                                        (
                                            "multiply_add",
                                            v_val[1],
                                            v_val[1],
                                            c_factor,
                                            c_const,
                                        ),
                                        (
                                            "multiply_add",
                                            v_val[2],
                                            v_val[2],
                                            c_factor,
                                            c_const,
                                        ),
                                    ],
                                    **({"load": loads} if loads else {}),
                                }
                            )
                        else:  # non-fusable stage
                            c1, c3 = hc[0], hc[1]
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
                            self.emit(
                                {
                                    "valu": [
                                        (op1, v_tmp1[0], v_val[0], c1),
                                        (op1, v_tmp1[1], v_val[1], c1),
                                        (op1, v_tmp1[2], v_val[2], c1),
                                        (op3, v_tmp2[0], v_val[0], c3),
                                        (op3, v_tmp2[1], v_val[1], c3),
                                        (op3, v_tmp2[2], v_val[2], c3),
                                    ],
                                    **({"load": loads} if loads else {}),
                                }
                            )
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
                            self.emit(
                                {
                                    "valu": [
                                        (op2, v_val[0], v_tmp1[0], v_tmp2[0]),
                                        (op2, v_val[1], v_tmp1[1], v_tmp2[1]),
                                        (op2, v_val[2], v_tmp1[2], v_tmp2[2]),
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
                            self.emit({"load": loads})

                    self.emit(
                        {
                            "valu": [
                                ("&", v_tmp1[0], v_val[0], v_one),
                                ("&", v_tmp1[1], v_val[1], v_one),
                                ("&", v_tmp1[2], v_val[2], v_one),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", v_tmp2[0], v_one, v_tmp1[0]),
                                ("+", v_tmp2[1], v_one, v_tmp1[1]),
                                ("+", v_tmp2[2], v_one, v_tmp1[2]),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_idx[0],
                                    s_idx + offs[0],
                                    v_two,
                                    v_tmp2[0],
                                ),
                                (
                                    "multiply_add",
                                    v_idx[1],
                                    s_idx + offs[1],
                                    v_two,
                                    v_tmp2[1],
                                ),
                                (
                                    "multiply_add",
                                    v_idx[2],
                                    s_idx + offs[2],
                                    v_two,
                                    v_tmp2[2],
                                ),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("<", v_tmp1[0], v_idx[0], v_n_nodes),
                                ("<", v_tmp1[1], v_idx[1], v_n_nodes),
                                ("<", v_tmp1[2], v_idx[2], v_n_nodes),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("*", v_idx[0], v_idx[0], v_tmp1[0]),
                                ("*", v_idx[1], v_idx[1], v_tmp1[1]),
                                ("*", v_idx[2], v_idx[2], v_tmp1[2]),
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", s_idx + offs[0], v_idx[0], v_zero),
                                ("+", s_idx + offs[1], v_idx[1], v_zero),
                                ("+", s_idx + offs[2], v_idx[2], v_zero),
                                ("+", s_val + offs[0], v_val[0], v_zero),
                                ("+", s_val + offs[1], v_val[1], v_zero),
                                ("+", s_val + offs[2], v_val[2], v_zero),
                            ]
                        }
                    )

                    if has_next:
                        v_node, v_node_next = v_node_next, v_node
                        v_addr, v_addr_next = v_addr_next, v_addr

                if remainder > 0:
                    offs = [(n_triples * 3 + i) * VLEN for i in range(remainder)]
                    if n_triples == 0:
                        self.emit(
                            {
                                "valu": [
                                    ("+", v_addr[i], v_forest_p, s_idx + offs[i])
                                    for i in range(remainder)
                                ]
                            }
                        )
                        for b in range(remainder):
                            for i in range(0, VLEN, 2):
                                if i + 1 < VLEN:
                                    self.emit(
                                        {
                                            "load": [
                                                (
                                                    "load_offset",
                                                    v_node[b],
                                                    v_addr[b],
                                                    i,
                                                ),
                                                (
                                                    "load_offset",
                                                    v_node[b],
                                                    v_addr[b],
                                                    i + 1,
                                                ),
                                            ]
                                        }
                                    )
                                else:
                                    self.emit(
                                        {
                                            "load": [
                                                ("load_offset", v_node[b], v_addr[b], i)
                                            ]
                                        }
                                    )

                    self.emit(
                        {
                            "valu": [
                                ("^", v_val[i], s_val + offs[i], v_node[i])
                                for i in range(remainder)
                            ]
                        }
                    )

                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        hc = hash_consts[hi]
                        if hc[2]:  # fusable
                            c_factor, c_const = hc[0], hc[1]
                            self.emit(
                                {
                                    "valu": [
                                        (
                                            "multiply_add",
                                            v_val[i],
                                            v_val[i],
                                            c_factor,
                                            c_const,
                                        )
                                        for i in range(remainder)
                                    ]
                                }
                            )
                        else:
                            c1, c3 = hc[0], hc[1]
                            ops1 = [
                                (op1, v_tmp1[i], v_val[i], c1) for i in range(remainder)
                            ]
                            ops1 += [
                                (op3, v_tmp2[i], v_val[i], c3) for i in range(remainder)
                            ]
                            self.emit({"valu": ops1})
                            self.emit(
                                {
                                    "valu": [
                                        (op2, v_val[i], v_tmp1[i], v_tmp2[i])
                                        for i in range(remainder)
                                    ]
                                }
                            )

                    self.emit(
                        {
                            "valu": [
                                ("&", v_tmp1[i], v_val[i], v_one)
                                for i in range(remainder)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("+", v_tmp2[i], v_one, v_tmp1[i])
                                for i in range(remainder)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                (
                                    "multiply_add",
                                    v_idx[i],
                                    s_idx + offs[i],
                                    v_two,
                                    v_tmp2[i],
                                )
                                for i in range(remainder)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("<", v_tmp1[i], v_idx[i], v_n_nodes)
                                for i in range(remainder)
                            ]
                        }
                    )
                    self.emit(
                        {
                            "valu": [
                                ("*", v_idx[i], v_idx[i], v_tmp1[i])
                                for i in range(remainder)
                            ]
                        }
                    )
                    wb = [
                        ("+", s_idx + offs[i], v_idx[i], v_zero)
                        for i in range(remainder)
                    ]
                    wb += [
                        ("+", s_val + offs[i], v_val[i], v_zero)
                        for i in range(remainder)
                    ]
                    self.emit({"valu": wb})

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
