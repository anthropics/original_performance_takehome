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

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        """
        Aggressively pipelined kernel: interleave A and B to overlap load/valu.
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Allocate and load header variables
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

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vlen_const = self.scratch_const(VLEN)
        vlen2_const = self.scratch_const(2 * VLEN)

        # Vector registers for batch A
        v_idx_A = self.alloc_vec("v_idx_A")
        v_val_A = self.alloc_vec("v_val_A")
        v_node_val_A = self.alloc_vec("v_node_val_A")
        v_tmp_A1 = self.alloc_vec("v_tmp_A1")
        v_tmp_A2 = self.alloc_vec("v_tmp_A2")

        # Vector registers for batch B
        v_idx_B = self.alloc_vec("v_idx_B")
        v_val_B = self.alloc_vec("v_val_B")
        v_node_val_B = self.alloc_vec("v_node_val_B")
        v_tmp_B1 = self.alloc_vec("v_tmp_B1")
        v_tmp_B2 = self.alloc_vec("v_tmp_B2")

        # Vector constants
        v_zero = self.alloc_vec("v_zero")
        v_one = self.alloc_vec("v_one")
        v_two = self.alloc_vec("v_two")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        v_fvp = self.alloc_vec("v_fvp")

        # Scalar tree addresses for A and B
        tree_addrs_A = [self.alloc_scratch(f"ta_A{i}") for i in range(VLEN)]
        tree_addrs_B = [self.alloc_scratch(f"ta_B{i}") for i in range(VLEN)]

        # Loop counters and base pointers
        round_ctr = self.alloc_scratch("round_ctr")
        batch_ctr = self.alloc_scratch("batch_ctr")
        idx_base_A = self.alloc_scratch("idx_base_A")
        val_base_A = self.alloc_scratch("val_base_A")
        idx_base_B = self.alloc_scratch("idx_base_B")
        val_base_B = self.alloc_scratch("val_base_B")

        # Hash constant vectors
        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1 = self.alloc_vec(f"hc1_{hi}")
            vc3 = self.alloc_vec(f"hc3_{hi}")
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            hash_consts.append((vc1, vc3, op1, op2, op3, c1, c3))

        # Broadcast hash constants (6 per cycle)
        for i in range(0, len(hash_consts), 3):
            slots = []
            for j in range(i, min(i + 3, len(hash_consts))):
                vc1, vc3, _, _, _, c1, c3 = hash_consts[j]
                slots.append(("vbroadcast", vc1, c1))
                slots.append(("vbroadcast", vc3, c3))
            self.instrs.append({"valu": slots})

        # Broadcast basic vector constants
        self.instrs.append(
            {
                "valu": [
                    ("vbroadcast", v_zero, zero_const),
                    ("vbroadcast", v_one, one_const),
                    ("vbroadcast", v_two, two_const),
                    ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
                    ("vbroadcast", v_fvp, self.scratch["forest_values_p"]),
                ]
            }
        )

        self.add("flow", ("pause",))
        self.add("load", ("const", round_ctr, 0))

        # === Outer loop (rounds) ===
        outer_loop_start = len(self.instrs)

        # Reset counters and bases
        self.instrs.append(
            {
                "load": [("const", batch_ctr, 0)],
                "alu": [
                    (
                        "+",
                        idx_base_A,
                        self.scratch["inp_indices_p"],
                        zero_const,
                    ),
                    (
                        "+",
                        val_base_A,
                        self.scratch["inp_values_p"],
                        zero_const,
                    ),
                ],
            }
        )
        self.instrs.append(
            {
                "alu": [
                    ("+", idx_base_B, idx_base_A, vlen_const),
                    ("+", val_base_B, val_base_A, vlen_const),
                ]
            }
        )

        # === Inner loop: interleaved A and B ===
        inner_loop_start = len(self.instrs)

        # PHASE 1: Load A idx/val (2 loads)
        self.instrs.append(
            {
                "load": [
                    ("vload", v_idx_A, idx_base_A),
                    ("vload", v_val_A, val_base_A),
                ]
            }
        )

        # PHASE 2: Compute A addresses (valu)
        self.add("valu", ("+", v_tmp_A1, v_idx_A, v_fvp))

        # PHASE 3: Copy A addresses (alu, 8 ops)
        self.instrs.append(
            {
                "alu": [
                    ("+", tree_addrs_A[i], v_tmp_A1 + i, zero_const)
                    for i in range(VLEN)
                ]
            }
        )

        # PHASE 4-7: Load A tree values (4 cycles)
        for i in range(0, VLEN, 2):
            self.instrs.append(
                {
                    "load": [
                        ("load", v_node_val_A + i, tree_addrs_A[i]),
                        ("load", v_node_val_A + i + 1, tree_addrs_A[i + 1]),
                    ]
                }
            )

        # PHASE 8: A XOR
        self.add("valu", ("^", v_val_A, v_val_A, v_node_val_A))

        # PHASE 9-20: A hash (12 cycles) - interleave with B setup
        # Hash stage 0 + B load
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[0]
        self.instrs.append(
            {
                "valu": [
                    (op1, v_tmp_A1, v_val_A, vc1),
                    (op3, v_tmp_A2, v_val_A, vc3),
                ],
                "load": [
                    ("vload", v_idx_B, idx_base_B),
                    ("vload", v_val_B, val_base_B),
                ],
            }
        )
        self.add("valu", (op2, v_val_A, v_tmp_A1, v_tmp_A2))

        # Hash stage 1 + B address compute
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[1]
        self.instrs.append(
            {
                "valu": [
                    (op1, v_tmp_A1, v_val_A, vc1),
                    (op3, v_tmp_A2, v_val_A, vc3),
                    ("+", v_tmp_B1, v_idx_B, v_fvp),
                ]
            }
        )
        self.add("valu", (op2, v_val_A, v_tmp_A1, v_tmp_A2))

        # Hash stage 2 + B address copy
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[2]
        self.instrs.append(
            {
                "valu": [
                    (op1, v_tmp_A1, v_val_A, vc1),
                    (op3, v_tmp_A2, v_val_A, vc3),
                ],
                "alu": [
                    ("+", tree_addrs_B[i], v_tmp_B1 + i, zero_const)
                    for i in range(VLEN)
                ],
            }
        )
        self.add("valu", (op2, v_val_A, v_tmp_A1, v_tmp_A2))

        # Hash stage 3 + B tree loads [0:2]
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[3]
        self.instrs.append(
            {
                "valu": [
                    (op1, v_tmp_A1, v_val_A, vc1),
                    (op3, v_tmp_A2, v_val_A, vc3),
                ],
                "load": [
                    ("load", v_node_val_B + 0, tree_addrs_B[0]),
                    ("load", v_node_val_B + 1, tree_addrs_B[1]),
                ],
            }
        )
        self.instrs.append(
            {
                "valu": [(op2, v_val_A, v_tmp_A1, v_tmp_A2)],
                "load": [
                    ("load", v_node_val_B + 2, tree_addrs_B[2]),
                    ("load", v_node_val_B + 3, tree_addrs_B[3]),
                ],
            }
        )

        # Hash stage 4 + B tree loads [4:6]
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[4]
        self.instrs.append(
            {
                "valu": [
                    (op1, v_tmp_A1, v_val_A, vc1),
                    (op3, v_tmp_A2, v_val_A, vc3),
                ],
                "load": [
                    ("load", v_node_val_B + 4, tree_addrs_B[4]),
                    ("load", v_node_val_B + 5, tree_addrs_B[5]),
                ],
            }
        )
        self.instrs.append(
            {
                "valu": [(op2, v_val_A, v_tmp_A1, v_tmp_A2)],
                "load": [
                    ("load", v_node_val_B + 6, tree_addrs_B[6]),
                    ("load", v_node_val_B + 7, tree_addrs_B[7]),
                ],
            }
        )

        # Hash stage 5
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[5]
        self.instrs.append(
            {
                "valu": [
                    (op1, v_tmp_A1, v_val_A, vc1),
                    (op3, v_tmp_A2, v_val_A, vc3),
                ]
            }
        )
        self.add("valu", (op2, v_val_A, v_tmp_A1, v_tmp_A2))

        # PHASE 21-26: A index update + B XOR + B early hash
        self.instrs.append(
            {
                "valu": [
                    ("&", v_tmp_A1, v_val_A, v_one),
                    ("*", v_idx_A, v_idx_A, v_two),
                    ("^", v_val_B, v_val_B, v_node_val_B),
                ]
            }
        )
        self.add("valu", ("==", v_tmp_A1, v_tmp_A1, v_zero))

        # A vselect + B hash 0 first
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[0]
        self.instrs.append(
            {
                "flow": [("vselect", v_tmp_A2, v_tmp_A1, v_one, v_two)],
                "valu": [
                    (op1, v_tmp_B1, v_val_B, vc1),
                    (op3, v_tmp_B2, v_val_B, vc3),
                ],
            }
        )

        # A (+) + B hash 0 second
        self.instrs.append(
            {
                "valu": [
                    ("+", v_idx_A, v_idx_A, v_tmp_A2),
                    (op2, v_val_B, v_tmp_B1, v_tmp_B2),
                ]
            }
        )

        # A (<) + B hash 1 first
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[1]
        self.instrs.append(
            {
                "valu": [
                    ("<", v_tmp_A1, v_idx_A, v_n_nodes),
                    (op1, v_tmp_B1, v_val_B, vc1),
                    (op3, v_tmp_B2, v_val_B, vc3),
                ]
            }
        )

        # A final vselect + B hash 1 second
        self.instrs.append(
            {
                "flow": [("vselect", v_idx_A, v_tmp_A1, v_idx_A, v_zero)],
                "valu": [(op2, v_val_B, v_tmp_B1, v_tmp_B2)],
            }
        )

        # PHASE 27: Store A + B hash 2
        vc1, vc3, op1, op2, op3, _, _ = hash_consts[2]
        self.instrs.append(
            {
                "store": [
                    ("vstore", idx_base_A, v_idx_A),
                    ("vstore", val_base_A, v_val_A),
                ],
                "valu": [
                    (op1, v_tmp_B1, v_val_B, vc1),
                    (op3, v_tmp_B2, v_val_B, vc3),
                ],
            }
        )
        self.add("valu", (op2, v_val_B, v_tmp_B1, v_tmp_B2))

        # PHASE 28-39: B hash stages 3-5 (6 cycles)
        for vc1, vc3, op1, op2, op3, _, _ in hash_consts[3:]:
            self.instrs.append(
                {
                    "valu": [
                        (op1, v_tmp_B1, v_val_B, vc1),
                        (op3, v_tmp_B2, v_val_B, vc3),
                    ]
                }
            )
            self.add("valu", (op2, v_val_B, v_tmp_B1, v_tmp_B2))

        # PHASE 40-45: B index update
        self.instrs.append(
            {
                "valu": [
                    ("&", v_tmp_B1, v_val_B, v_one),
                    ("*", v_idx_B, v_idx_B, v_two),
                ]
            }
        )
        self.add("valu", ("==", v_tmp_B1, v_tmp_B1, v_zero))
        self.add("flow", ("vselect", v_tmp_B2, v_tmp_B1, v_one, v_two))
        self.add("valu", ("+", v_idx_B, v_idx_B, v_tmp_B2))
        self.add("valu", ("<", v_tmp_B1, v_idx_B, v_n_nodes))
        self.add("flow", ("vselect", v_idx_B, v_tmp_B1, v_idx_B, v_zero))

        # Store B + update counters
        self.instrs.append(
            {
                "store": [
                    ("vstore", idx_base_B, v_idx_B),
                    ("vstore", val_base_B, v_val_B),
                ],
                "alu": [
                    ("+", batch_ctr, batch_ctr, vlen2_const),
                    ("+", idx_base_A, idx_base_A, vlen2_const),
                    ("+", val_base_A, val_base_A, vlen2_const),
                    ("+", idx_base_B, idx_base_B, vlen2_const),
                    ("+", val_base_B, val_base_B, vlen2_const),
                ],
            }
        )

        # Inner loop check
        self.add("alu", ("<", tmp1, batch_ctr, self.scratch["batch_size"]))
        self.add("flow", ("cond_jump", tmp1, inner_loop_start))

        # Outer loop update
        self.add("flow", ("add_imm", round_ctr, round_ctr, 1))
        self.add("alu", ("<", tmp1, round_ctr, self.scratch["rounds"]))
        self.add("flow", ("cond_jump", tmp1, outer_loop_start))

        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(
    forest_height, rounds, batch_size, seed=123, trace=False, prints=False
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(
        forest.height, len(forest.values), len(inp.indices), rounds
    )

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
            print(
                machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
            )
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
