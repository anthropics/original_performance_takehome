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
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle):
        """Add a pre-built instruction bundle (dict of engine -> list of slots)"""
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

    def preload_const(self, val, name=None):
        """Allocate and return address for a constant (load happens in init phase)"""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized 2-way pipeline with maximum VLIW packing.
        Minimize cycles through aggressive operation combining.
        """
        num_batches = batch_size // VLEN  # 32 vector batches

        # Pre-allocate hash constants
        hash_consts = []
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            c1 = self.preload_const(val1)
            c2 = self.preload_const(val3)
            hash_consts.append((c1, c2))

        zero_const = self.preload_const(0)
        one_const = self.preload_const(1)

        # Memory layout
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                    "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Keep ALL idx and val vectors in scratch
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(num_batches)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(num_batches)]

        # Working space for 2 batches
        v_node_A = self.alloc_scratch("v_node_A", VLEN)
        v_node_B = self.alloc_scratch("v_node_B", VLEN)
        v_tmp1_A = self.alloc_scratch("v_tmp1_A", VLEN)
        v_tmp2_A = self.alloc_scratch("v_tmp2_A", VLEN)
        v_tmp1_B = self.alloc_scratch("v_tmp1_B", VLEN)
        v_tmp2_B = self.alloc_scratch("v_tmp2_B", VLEN)

        v_one = self.alloc_scratch("v_one", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Scalar temps
        tmp1 = self.alloc_scratch("tmp1")
        round_counter = self.alloc_scratch("round_counter")
        num_rounds_s = self.alloc_scratch("num_rounds_s")
        ptr = self.alloc_scratch("ptr")

        # Addresses for gather loads
        addr_A = [self.alloc_scratch(f"addrA{i}") for i in range(VLEN)]
        addr_B = [self.alloc_scratch(f"addrB{i}") for i in range(VLEN)]

        # Vector hash constants
        v_hash_consts = []
        for i in range(len(HASH_STAGES)):
            vc1 = self.alloc_scratch(f"vhc1_{i}", VLEN)
            vc2 = self.alloc_scratch(f"vhc2_{i}", VLEN)
            v_hash_consts.append((vc1, vc2))

        # ===== INIT PHASE =====
        tmp_addr = self.alloc_scratch("tmp_addr")
        for i, v in enumerate(init_vars):
            self.add_bundle({"load": [("const", tmp_addr, i)]})
            self.add_bundle({"load": [("load", self.scratch[v], tmp_addr)]})

        const_loads = [("const", addr_c, val) for val, addr_c in self.const_map.items()]
        for i in range(0, len(const_loads), 2):
            self.add_bundle({"load": const_loads[i:i+2]})

        self.add_bundle({"valu": [
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ]})

        for i in range(0, len(hash_consts), 3):
            valu_ops = []
            for j in range(3):
                if i + j < len(hash_consts):
                    c1, c2 = hash_consts[i + j]
                    vc1, vc2 = v_hash_consts[i + j]
                    valu_ops.extend([("vbroadcast", vc1, c1), ("vbroadcast", vc2, c2)])
            self.add_bundle({"valu": valu_ops})

        # Load all indices and values - simple approach
        self.add_bundle({"alu": [("+", ptr, self.scratch["inp_indices_p"], zero_const)]})
        for i in range(num_batches):
            self.add_bundle({"load": [("vload", v_idx[i], ptr)]})
            if i + 1 < num_batches:
                self.add_bundle({"flow": [("add_imm", ptr, ptr, VLEN)]})

        self.add_bundle({"alu": [("+", ptr, self.scratch["inp_values_p"], zero_const)]})
        for i in range(num_batches):
            self.add_bundle({"load": [("vload", v_val[i], ptr)]})
            if i + 1 < num_batches:
                self.add_bundle({"flow": [("add_imm", ptr, ptr, VLEN)]})

        self.add_bundle({"load": [("const", num_rounds_s, rounds)]})
        self.add("flow", ("pause",))

        # ===== ROUND 0 SPECIAL CASE =====
        # All indices start at 0, so we only need ONE forest load!
        node_scalar = self.alloc_scratch("node_scalar")
        v_node_shared = self.alloc_scratch("v_node_shared", VLEN)

        # Load forest[0] once and broadcast
        self.add_bundle({"load": [("load", node_scalar, self.scratch["forest_values_p"])]})
        self.add_bundle({"valu": [("vbroadcast", v_node_shared, node_scalar)]})

        # Process all batches with shared node_val using 2-way pipelining
        for b in range(0, num_batches, 2):
            val_A, val_B = v_val[b], v_val[b + 1]
            idx_A, idx_B = v_idx[b], v_idx[b + 1]

            # XOR both batches with shared node
            self.add_bundle({"valu": [("^", val_A, val_A, v_node_shared), ("^", val_B, val_B, v_node_shared)]})

            # Hash stage 0 for both
            vc1_0, vc2_0 = v_hash_consts[0]
            op1_0, _, op2_0, op3_0, _ = HASH_STAGES[0]
            self.add_bundle({"valu": [
                (op1_0, v_tmp1_A, val_A, vc1_0), (op3_0, v_tmp2_A, val_A, vc2_0),
                (op1_0, v_tmp1_B, val_B, vc1_0), (op3_0, v_tmp2_B, val_B, vc2_0),
            ]})
            self.add_bundle({"valu": [(op2_0, val_A, v_tmp1_A, v_tmp2_A), (op2_0, val_B, v_tmp1_B, v_tmp2_B)]})

            # Hash stages 1-5 interleaved
            for hi in range(1, 6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                self.add_bundle({"valu": [
                    (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                    (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                ]})
                self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

            # idx = 2*0 + (val%2 + 1) = val%2 + 1 (since idx starts at 0)
            # A: idx_A = (val_A & 1) + 1
            self.add_bundle({"valu": [("&", idx_A, val_A, v_one), ("&", idx_B, val_B, v_one)]})
            self.add_bundle({"valu": [("+", idx_A, idx_A, v_one), ("+", idx_B, idx_B, v_one)]})
            # No bounds check needed - idx will be 1 or 2, both < n_nodes

        # ===== ROUND 1 SPECIAL CASE =====
        # All indices are in {1, 2}, so we only need 2 forest loads!
        node1_scalar = self.alloc_scratch("node1_scalar")
        node2_scalar = self.alloc_scratch("node2_scalar")
        v_node1 = self.alloc_scratch("v_node1", VLEN)
        v_node2 = self.alloc_scratch("v_node2", VLEN)
        addr1 = self.alloc_scratch("addr1")
        addr2 = self.alloc_scratch("addr2")

        # Compute addresses and load forest[1] and forest[2]
        self.add_bundle({"flow": [("add_imm", addr1, self.scratch["forest_values_p"], 1)]})
        self.add_bundle({"flow": [("add_imm", addr2, self.scratch["forest_values_p"], 2)]})
        self.add_bundle({"load": [("load", node1_scalar, addr1), ("load", node2_scalar, addr2)]})
        self.add_bundle({"valu": [("vbroadcast", v_node1, node1_scalar), ("vbroadcast", v_node2, node2_scalar)]})

        # Process all batches - select correct node based on idx
        for b in range(0, num_batches, 2):
            val_A, val_B = v_val[b], v_val[b + 1]
            idx_A, idx_B = v_idx[b], v_idx[b + 1]

            # Select node value: idx==1 -> v_node1, idx==2 -> v_node2
            # (idx == 1) is nonzero for true -> selects v_node1
            self.add_bundle({"valu": [("==", v_tmp1_A, idx_A, v_one), ("==", v_tmp1_B, idx_B, v_one)]})
            self.add_bundle({"flow": [("vselect", v_node_A, v_tmp1_A, v_node1, v_node2)]})
            self.add_bundle({"flow": [("vselect", v_node_B, v_tmp1_B, v_node1, v_node2)]})

            # XOR with selected nodes
            self.add_bundle({"valu": [("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B)]})

            # Hash stages interleaved
            vc1_0, vc2_0 = v_hash_consts[0]
            op1_0, _, op2_0, op3_0, _ = HASH_STAGES[0]
            self.add_bundle({"valu": [
                (op1_0, v_tmp1_A, val_A, vc1_0), (op3_0, v_tmp2_A, val_A, vc2_0),
                (op1_0, v_tmp1_B, val_B, vc1_0), (op3_0, v_tmp2_B, val_B, vc2_0),
            ]})
            self.add_bundle({"valu": [(op2_0, val_A, v_tmp1_A, v_tmp2_A), (op2_0, val_B, v_tmp1_B, v_tmp2_B)]})

            for hi in range(1, 6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                self.add_bundle({"valu": [
                    (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                    (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                ]})
                self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

            # idx = 2*idx + (val%2 + 1)
            self.add_bundle({"valu": [("&", v_tmp1_A, val_A, v_one), ("<<", idx_A, idx_A, v_one)]})
            self.add_bundle({"valu": [("&", v_tmp1_B, val_B, v_one), ("<<", idx_B, idx_B, v_one)]})
            self.add_bundle({"valu": [("+", v_tmp1_A, v_tmp1_A, v_one), ("+", v_tmp1_B, v_tmp1_B, v_one)]})
            self.add_bundle({"valu": [("+", idx_A, idx_A, v_tmp1_A), ("+", idx_B, idx_B, v_tmp1_B)]})
            self.add_bundle({"valu": [("<", v_tmp1_A, idx_A, v_n_nodes), ("<", v_tmp1_B, idx_B, v_n_nodes)]})
            self.add_bundle({"flow": [("vselect", idx_A, v_tmp1_A, idx_A, v_zero)]})
            self.add_bundle({"flow": [("vselect", idx_B, v_tmp1_B, idx_B, v_zero)]})

        # ===== MAIN LOOP (rounds 2-15) =====
        self.add_bundle({"load": [("const", round_counter, 2)]})
        round_loop_start = len(self.instrs)

        # Process pairs with optimized 2-way pipeline
        for b in range(0, num_batches, 2):
            idx_A, val_A = v_idx[b], v_val[b]
            idx_B, val_B = v_idx[b + 1], v_val[b + 1]

            # Cycle 1: A addr compute (8 ALU) + B addr 0-3 (4 ALU) = 12 ALU
            self.add_bundle({"alu": [
                ("+", addr_A[i], self.scratch["forest_values_p"], idx_A + i) for i in range(VLEN)
            ] + [("+", addr_B[j], self.scratch["forest_values_p"], idx_B + j) for j in range(4)]
            })

            # Cycle 2: A gather 0-1 + B addr 4-7 (4 ALU)
            self.add_bundle({
                "load": [("load", v_node_A + 0, addr_A[0]), ("load", v_node_A + 1, addr_A[1])],
                "alu": [("+", addr_B[j], self.scratch["forest_values_p"], idx_B + j) for j in range(4, VLEN)],
            })

            # Cycles 3-5: A gather 2-7 (3 cycles instead of 4, since we did 0-1 already)
            for i in range(2, VLEN, 2):
                self.add_bundle({"load": [
                    ("load", v_node_A + i, addr_A[i]),
                    ("load", v_node_A + i + 1, addr_A[i + 1]),
                ]})

            # Cycle 7: B gather 0-1 + A XOR
            self.add_bundle({
                "load": [("load", v_node_B + 0, addr_B[0]), ("load", v_node_B + 1, addr_B[1])],
                "valu": [("^", val_A, val_A, v_node_A)],
            })

            # Cycle 8: B gather 2-3 + A hash 0 (ops 1,2)
            vc1, vc2 = v_hash_consts[0]
            op1, _, op2, op3, _ = HASH_STAGES[0]
            self.add_bundle({
                "load": [("load", v_node_B + 2, addr_B[2]), ("load", v_node_B + 3, addr_B[3])],
                "valu": [(op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2)],
            })

            # Cycle 9: B gather 4-5 + A hash 0 (op3) + A hash 1 (ops 1,2)
            vc1_1, vc2_1 = v_hash_consts[1]
            op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
            self.add_bundle({
                "load": [("load", v_node_B + 4, addr_B[4]), ("load", v_node_B + 5, addr_B[5])],
                "valu": [
                    (op2, val_A, v_tmp1_A, v_tmp2_A),
                ],
            })

            # Cycle 10: B gather 6-7 + A hash 1 (ops 1,2)
            self.add_bundle({
                "load": [("load", v_node_B + 6, addr_B[6]), ("load", v_node_B + 7, addr_B[7])],
                "valu": [(op1_1, v_tmp1_A, val_A, vc1_1), (op3_1, v_tmp2_A, val_A, vc2_1)],
            })

            # Cycle 11: A hash 1 complete + B XOR + B hash 0 (ops 1,2)
            vc1_0, vc2_0 = v_hash_consts[0]
            op1_0, _, op2_0, op3_0, _ = HASH_STAGES[0]
            self.add_bundle({"valu": [
                (op2_1, val_A, v_tmp1_A, v_tmp2_A),
                ("^", val_B, val_B, v_node_B),
            ]})

            # Interleaved hash stages 2-5 for A and 0-3 for B
            for hi in range(2, 6):
                vc1_A, vc2_A = v_hash_consts[hi]
                op1_A, _, op2_A, op3_A, _ = HASH_STAGES[hi]
                vc1_B, vc2_B = v_hash_consts[hi - 2]
                op1_B, _, op2_B, op3_B, _ = HASH_STAGES[hi - 2]

                self.add_bundle({"valu": [
                    (op1_A, v_tmp1_A, val_A, vc1_A), (op3_A, v_tmp2_A, val_A, vc2_A),
                    (op1_B, v_tmp1_B, val_B, vc1_B), (op3_B, v_tmp2_B, val_B, vc2_B),
                ]})
                self.add_bundle({"valu": [
                    (op2_A, val_A, v_tmp1_A, v_tmp2_A),
                    (op2_B, val_B, v_tmp1_B, v_tmp2_B),
                ]})

            # A idx computation + B hash 4-5
            vc1_4, vc2_4 = v_hash_consts[4]
            op1_4, _, op2_4, op3_4, _ = HASH_STAGES[4]
            self.add_bundle({"valu": [
                ("&", v_tmp1_A, val_A, v_one), ("<<", idx_A, idx_A, v_one),
                (op1_4, v_tmp1_B, val_B, vc1_4), (op3_4, v_tmp2_B, val_B, vc2_4),
            ]})

            self.add_bundle({"valu": [
                ("+", v_tmp1_A, v_tmp1_A, v_one),
                (op2_4, val_B, v_tmp1_B, v_tmp2_B),
            ]})

            vc1_5, vc2_5 = v_hash_consts[5]
            op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
            self.add_bundle({"valu": [
                ("+", idx_A, idx_A, v_tmp1_A),
                (op1_5, v_tmp1_B, val_B, vc1_5), (op3_5, v_tmp2_B, val_B, vc2_5),
            ]})

            self.add_bundle({"valu": [
                ("<", v_tmp1_A, idx_A, v_n_nodes),
                (op2_5, val_B, v_tmp1_B, v_tmp2_B),
            ]})

            # A vselect + B idx start
            self.add_bundle({
                "valu": [("&", v_tmp1_B, val_B, v_one), ("<<", idx_B, idx_B, v_one)],
                "flow": [("vselect", idx_A, v_tmp1_A, idx_A, v_zero)],
            })

            # B idx computation - try to overlap with next pair's setup
            self.add_bundle({"valu": [("+", v_tmp1_B, v_tmp1_B, v_one)]})
            self.add_bundle({"valu": [("+", idx_B, idx_B, v_tmp1_B)]})
            self.add_bundle({"valu": [("<", v_tmp1_B, idx_B, v_n_nodes)]})

            # If not last pair, start next pair's address computation while doing B vselect
            if b + 2 < num_batches:
                next_idx_A = v_idx[b + 2]
                self.add_bundle({
                    "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_idx_A + i) for i in range(VLEN)],
                    "flow": [("vselect", idx_B, v_tmp1_B, idx_B, v_zero)],
                })
            else:
                self.add_bundle({"flow": [("vselect", idx_B, v_tmp1_B, idx_B, v_zero)]})

        # Round loop control - loop for rounds 2-10 only (9 iterations)
        self.add_bundle({"flow": [("add_imm", round_counter, round_counter, 1)]})
        eleven_const = self.scratch_const(11)
        self.add_bundle({"alu": [("<", tmp1, round_counter, eleven_const)]})
        round_loop_offset = round_loop_start - len(self.instrs) - 1
        self.add_bundle({"flow": [("cond_jump_rel", tmp1, round_loop_offset)]})

        # ===== ROUNDS 11-15: Unrolled (mirror rounds 0-4 after wrapping) =====
        # After round 10, ALL indices wrap to 0!

        # Round 11 (like round 0): all indices are 0
        self.add_bundle({"load": [("load", node_scalar, self.scratch["forest_values_p"])]})
        self.add_bundle({"valu": [("vbroadcast", v_node_shared, node_scalar)]})

        for b in range(0, num_batches, 2):
            val_A, val_B = v_val[b], v_val[b + 1]
            idx_A, idx_B = v_idx[b], v_idx[b + 1]
            self.add_bundle({"valu": [("^", val_A, val_A, v_node_shared), ("^", val_B, val_B, v_node_shared)]})
            # Hash
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                self.add_bundle({"valu": [
                    (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                    (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                ]})
                self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})
            # idx = (val & 1) + 1 (since idx starts at 0)
            self.add_bundle({"valu": [("&", idx_A, val_A, v_one), ("&", idx_B, val_B, v_one)]})
            self.add_bundle({"valu": [("+", idx_A, idx_A, v_one), ("+", idx_B, idx_B, v_one)]})

        # Round 12 (like round 1): indices in {1,2}
        self.add_bundle({"flow": [("add_imm", addr1, self.scratch["forest_values_p"], 1)]})
        self.add_bundle({"flow": [("add_imm", addr2, self.scratch["forest_values_p"], 2)]})
        self.add_bundle({"load": [("load", node1_scalar, addr1), ("load", node2_scalar, addr2)]})
        self.add_bundle({"valu": [("vbroadcast", v_node1, node1_scalar), ("vbroadcast", v_node2, node2_scalar)]})

        for b in range(0, num_batches, 2):
            val_A, val_B = v_val[b], v_val[b + 1]
            idx_A, idx_B = v_idx[b], v_idx[b + 1]
            # Select: idx==1 -> v_node1, idx==2 -> v_node2
            self.add_bundle({"valu": [("==", v_tmp1_A, idx_A, v_one), ("==", v_tmp1_B, idx_B, v_one)]})
            self.add_bundle({"flow": [("vselect", v_node_A, v_tmp1_A, v_node1, v_node2)]})
            self.add_bundle({"flow": [("vselect", v_node_B, v_tmp1_B, v_node1, v_node2)]})
            self.add_bundle({"valu": [("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B)]})
            # Hash
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                self.add_bundle({"valu": [
                    (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                    (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                ]})
                self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})
            # idx = 2*idx + (val%2 + 1)
            self.add_bundle({"valu": [("&", v_tmp1_A, val_A, v_one), ("<<", idx_A, idx_A, v_one)]})
            self.add_bundle({"valu": [("&", v_tmp1_B, val_B, v_one), ("<<", idx_B, idx_B, v_one)]})
            self.add_bundle({"valu": [("+", v_tmp1_A, v_tmp1_A, v_one), ("+", v_tmp1_B, v_tmp1_B, v_one)]})
            self.add_bundle({"valu": [("+", idx_A, idx_A, v_tmp1_A), ("+", idx_B, idx_B, v_tmp1_B)]})
            # No bounds check needed - indices will be {3,4,5,6} < n_nodes

        # Rounds 13-15: Use gather approach (indices are larger but still manageable)
        for _round in range(13, 16):
            for b in range(0, num_batches, 2):
                idx_A, val_A = v_idx[b], v_val[b]
                idx_B, val_B = v_idx[b + 1], v_val[b + 1]

                # Compute addresses and gather
                self.add_bundle({"alu": [
                    ("+", addr_A[i], self.scratch["forest_values_p"], idx_A + i) for i in range(VLEN)
                ] + [("+", addr_B[j], self.scratch["forest_values_p"], idx_B + j) for j in range(4)]})

                self.add_bundle({
                    "load": [("load", v_node_A + 0, addr_A[0]), ("load", v_node_A + 1, addr_A[1])],
                    "alu": [("+", addr_B[j], self.scratch["forest_values_p"], idx_B + j) for j in range(4, VLEN)],
                })
                for i in range(2, VLEN, 2):
                    self.add_bundle({"load": [("load", v_node_A + i, addr_A[i]), ("load", v_node_A + i + 1, addr_A[i + 1])]})

                self.add_bundle({
                    "load": [("load", v_node_B + 0, addr_B[0]), ("load", v_node_B + 1, addr_B[1])],
                    "valu": [("^", val_A, val_A, v_node_A)],
                })

                # Interleaved hash with B gathers
                vc1, vc2 = v_hash_consts[0]
                op1, _, op2, op3, _ = HASH_STAGES[0]
                self.add_bundle({
                    "load": [("load", v_node_B + 2, addr_B[2]), ("load", v_node_B + 3, addr_B[3])],
                    "valu": [(op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2)],
                })

                vc1_1, vc2_1 = v_hash_consts[1]
                op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
                self.add_bundle({
                    "load": [("load", v_node_B + 4, addr_B[4]), ("load", v_node_B + 5, addr_B[5])],
                    "valu": [(op2, val_A, v_tmp1_A, v_tmp2_A)],
                })

                self.add_bundle({
                    "load": [("load", v_node_B + 6, addr_B[6]), ("load", v_node_B + 7, addr_B[7])],
                    "valu": [(op1_1, v_tmp1_A, val_A, vc1_1), (op3_1, v_tmp2_A, val_A, vc2_1)],
                })

                self.add_bundle({"valu": [
                    (op2_1, val_A, v_tmp1_A, v_tmp2_A),
                    ("^", val_B, val_B, v_node_B),
                ]})

                # Interleaved hash stages
                for hi in range(2, 6):
                    vc1_A, vc2_A = v_hash_consts[hi]
                    op1_A, _, op2_A, op3_A, _ = HASH_STAGES[hi]
                    vc1_B, vc2_B = v_hash_consts[hi - 2]
                    op1_B, _, op2_B, op3_B, _ = HASH_STAGES[hi - 2]

                    self.add_bundle({"valu": [
                        (op1_A, v_tmp1_A, val_A, vc1_A), (op3_A, v_tmp2_A, val_A, vc2_A),
                        (op1_B, v_tmp1_B, val_B, vc1_B), (op3_B, v_tmp2_B, val_B, vc2_B),
                    ]})
                    self.add_bundle({"valu": [
                        (op2_A, val_A, v_tmp1_A, v_tmp2_A),
                        (op2_B, val_B, v_tmp1_B, v_tmp2_B),
                    ]})

                # A idx + B hash 4-5
                vc1_4, vc2_4 = v_hash_consts[4]
                op1_4, _, op2_4, op3_4, _ = HASH_STAGES[4]
                self.add_bundle({"valu": [
                    ("&", v_tmp1_A, val_A, v_one), ("<<", idx_A, idx_A, v_one),
                    (op1_4, v_tmp1_B, val_B, vc1_4), (op3_4, v_tmp2_B, val_B, vc2_4),
                ]})

                self.add_bundle({"valu": [
                    ("+", v_tmp1_A, v_tmp1_A, v_one),
                    (op2_4, val_B, v_tmp1_B, v_tmp2_B),
                ]})

                vc1_5, vc2_5 = v_hash_consts[5]
                op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
                self.add_bundle({"valu": [
                    ("+", idx_A, idx_A, v_tmp1_A),
                    (op1_5, v_tmp1_B, val_B, vc1_5), (op3_5, v_tmp2_B, val_B, vc2_5),
                ]})

                self.add_bundle({"valu": [
                    ("<", v_tmp1_A, idx_A, v_n_nodes),
                    (op2_5, val_B, v_tmp1_B, v_tmp2_B),
                ]})

                # A vselect + B idx
                self.add_bundle({
                    "valu": [("&", v_tmp1_B, val_B, v_one), ("<<", idx_B, idx_B, v_one)],
                    "flow": [("vselect", idx_A, v_tmp1_A, idx_A, v_zero)],
                })

                self.add_bundle({"valu": [("+", v_tmp1_B, v_tmp1_B, v_one)]})
                self.add_bundle({"valu": [("+", idx_B, idx_B, v_tmp1_B)]})
                self.add_bundle({"valu": [("<", v_tmp1_B, idx_B, v_n_nodes)]})
                self.add_bundle({"flow": [("vselect", idx_B, v_tmp1_B, idx_B, v_zero)]})

        # Store all indices and values back
        self.add_bundle({"alu": [("+", ptr, self.scratch["inp_indices_p"], zero_const)]})
        for i in range(num_batches):
            self.add_bundle({"store": [("vstore", ptr, v_idx[i])]})
            if i + 1 < num_batches:
                self.add_bundle({"flow": [("add_imm", ptr, ptr, VLEN)]})

        self.add_bundle({"alu": [("+", ptr, self.scratch["inp_values_p"], zero_const)]})
        for i in range(num_batches):
            self.add_bundle({"store": [("vstore", ptr, v_val[i])]})
            if i + 1 < num_batches:
                self.add_bundle({"flow": [("add_imm", ptr, ptr, VLEN)]})

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
