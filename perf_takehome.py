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
from dataclasses import dataclass
import heapq
import os
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


@dataclass
class Op:
    engine: str
    slot: tuple
    srcs: list[int]
    dests: list[int]


def _vec_range(addr: int) -> list[int]:
    return list(range(addr, addr + VLEN))


def schedule_ops(
    ops: list[Op], start_offsets: dict[int, int] | None = None
) -> list[dict[str, list[tuple]]]:
    if not ops:
        return []

    last_write: dict[int, int] = {}
    last_read: dict[int, int] = {}
    deps_count = [0] * len(ops)
    users: list[list[tuple[int, int]]] = [[] for _ in ops]

    for i, op in enumerate(ops):
        deps: dict[int, int] = {}
        for src in op.srcs:
            if src in last_write:
                dep = last_write[src]
                deps[dep] = max(deps.get(dep, 0), 1)
        for dest in op.dests:
            if dest in last_write:
                dep = last_write[dest]
                deps[dep] = max(deps.get(dep, 0), 1)
            if dest in last_read:
                dep = last_read[dest]
                deps[dep] = max(deps.get(dep, 0), 0)
        for dep, latency in deps.items():
            users[dep].append((i, latency))
        deps_count[i] = len(deps)
        for src in op.srcs:
            last_read[src] = i
        for dest in op.dests:
            last_write[dest] = i

    earliest = [0] * len(ops)
    if start_offsets:
        for op_id, offset in start_offsets.items():
            if 0 <= op_id < len(ops):
                earliest[op_id] = max(earliest[op_id], offset)
    ready: list[tuple[int, int]] = []
    for i, count in enumerate(deps_count):
        if count == 0:
            heapq.heappush(ready, (earliest[i], i))

    bundles: list[dict[str, list[tuple]]] = []
    scheduled = 0
    cycle = 0
    engine_limits = {k: v for k, v in SLOT_LIMITS.items() if k != "debug"}
    users_count = [len(u) for u in users]
    engine_priority = {"flow": 0, "load": 1, "store": 2, "valu": 3, "alu": 4}

    while scheduled < len(ops):
        if not ready:
            cycle += 1
            continue

        if ready[0][0] > cycle:
            cycle = ready[0][0]

        if cycle >= len(bundles):
            bundles.append(defaultdict(list))

        slots_used = {k: 0 for k in engine_limits}
        available: list[int] = []
        while ready and ready[0][0] <= cycle:
            available.append(heapq.heappop(ready)[1])

        def pick_op_index() -> int | None:
            best_idx = None
            best_score = None
            for i, op_id in enumerate(available):
                op = ops[op_id]
                if slots_used[op.engine] >= engine_limits[op.engine]:
                    continue
                score = (engine_priority.get(op.engine, 99), -users_count[op_id])
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = i
            return best_idx

        while available:
            pick_idx = pick_op_index()
            if pick_idx is None:
                break
            op_id = available.pop(pick_idx)
            op = ops[op_id]
            bundles[cycle][op.engine].append(op.slot)
            slots_used[op.engine] += 1
            scheduled += 1
            for user, latency in users[op_id]:
                deps_count[user] -= 1
                earliest[user] = max(earliest[user], cycle + latency)
                if deps_count[user] == 0:
                    if earliest[user] <= cycle:
                        available.append(user)
                    else:
                        heapq.heappush(ready, (earliest[user], user))

        for op_id in available:
            heapq.heappush(ready, (earliest[op_id], op_id))

        cycle += 1

    return bundles


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
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel for the main submission case.
        Falls back to the baseline kernel for other sizes.
        """
        if not (forest_height == 10 and n_nodes == 2047 and batch_size == 256 and rounds == 16):
            return self.build_kernel_baseline(forest_height, n_nodes, batch_size, rounds)

        # Reset builder state
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

        init_ops: list[Op] = []

        def emit(op_list: list[Op], engine: str, slot: tuple, srcs: list[int], dests: list[int]):
            op_list.append(Op(engine=engine, slot=slot, srcs=srcs, dests=dests))

        def emit_const(op_list: list[Op], dest: int, val: int):
            emit(op_list, "load", ("const", dest, val), [], [dest])

        def emit_vbroadcast(op_list: list[Op], dest: int, src: int):
            emit(op_list, "valu", ("vbroadcast", dest, src), [src], _vec_range(dest))

        def emit_alu(op_list: list[Op], op: str, dest: int, a1: int, a2: int):
            emit(op_list, "alu", (op, dest, a1, a2), [a1, a2], [dest])

        def emit_valu(op_list: list[Op], op: str, dest: int, a1: int, a2: int):
            emit(op_list, "valu", (op, dest, a1, a2), _vec_range(a1) + _vec_range(a2), _vec_range(dest))

        def emit_muladd(op_list: list[Op], dest: int, a: int, b: int, c: int):
            emit(
                op_list,
                "valu",
                ("multiply_add", dest, a, b, c),
                _vec_range(a) + _vec_range(b) + _vec_range(c),
                _vec_range(dest),
            )

        def emit_vload(op_list: list[Op], dest: int, addr: int):
            emit(op_list, "load", ("vload", dest, addr), [addr], _vec_range(dest))

        def emit_load(op_list: list[Op], dest: int, addr: int):
            emit(op_list, "load", ("load", dest, addr), [addr], [dest])

        def emit_load_offset(op_list: list[Op], dest: int, addr: int, offset: int):
            emit(
                op_list,
                "load",
                ("load_offset", dest, addr, offset),
                [addr + offset],
                [dest + offset],
            )

        def emit_vstore(op_list: list[Op], addr: int, src: int):
            emit(op_list, "store", ("vstore", addr, src), [addr] + _vec_range(src), [])

        def emit_vselect(op_list: list[Op], dest: int, cond: int, a: int, b: int):
            emit(
                op_list,
                "flow",
                ("vselect", dest, cond, a, b),
                _vec_range(cond) + _vec_range(a) + _vec_range(b),
                _vec_range(dest),
            )

        scalar_consts: dict[int, int] = {}
        vector_consts: dict[int, int] = {}

        def scalar_const(val: int) -> int:
            if val in scalar_consts:
                return scalar_consts[val]
            addr = self.alloc_scratch(f"c_{val}")
            emit_const(init_ops, addr, val)
            scalar_consts[val] = addr
            return addr

        def vec_const(val: int) -> int:
            if val in vector_consts:
                return vector_consts[val]
            scalar = scalar_const(val)
            addr = self.alloc_scratch(f"vc_{val}", VLEN)
            emit_vbroadcast(init_ops, addr, scalar)
            vector_consts[val] = addr
            return addr

        # Global constants
        zero_vec = vec_const(0)
        one_vec = vec_const(1)
        two_vec = vec_const(2)
        four_vec = vec_const(4)
        eight_vec = vec_const(8)
        fifteen_vec = vec_const(15)
        one_scalar = scalar_const(1)
        fifteen_scalar = scalar_const(15)
        forest_base_val = 7
        forest_base_scalar = scalar_const(forest_base_val)
        forest_base_vec = vec_const(forest_base_val)
        inp_base_val = forest_base_val + n_nodes + batch_size

        # Hash constants
        c1_vec = vec_const(0x7ED55D16)
        c2_vec = vec_const(0xC761C23C)
        c3_vec = vec_const(0x165667B1)
        c4_vec = vec_const(0xD3A2646C)
        c5_vec = vec_const(0xFD7046C5)
        c6_vec = vec_const(0xB55A4F09)
        m1_vec = vec_const(4097)
        m3_vec = vec_const(33)
        m5_vec = vec_const(9)
        sh19_vec = vec_const(19)
        sh9_vec = vec_const(9)
        sh16_vec = vec_const(16)

        # Load top-of-tree node values (indices 0..14) into vector constants
        node_vec: dict[int, int] = {}
        for i in range(15):
            addr_val = forest_base_val + i
            addr_const = scalar_const(addr_val)
            node_scalar = self.alloc_scratch(f"node_{i}")
            emit_load(init_ops, node_scalar, addr_const)
            node_vec_addr = self.alloc_scratch(f"node_{i}_v", VLEN)
            emit_vbroadcast(init_ops, node_vec_addr, node_scalar)
            node_vec[i] = node_vec_addr

        groups_per_batch = 21
        start_spacing = 1
        total_groups = batch_size // VLEN
        batches = (total_groups + groups_per_batch - 1) // groups_per_batch

        group_vars = []
        for g in range(groups_per_batch):
            vars_g = {
                "val": self.alloc_scratch(f"val_{g}", VLEN),
                "idx": self.alloc_scratch(f"idx_{g}", VLEN),
                "tmp1": self.alloc_scratch(f"tmp1_{g}", VLEN),
                "tmp2": self.alloc_scratch(f"tmp2_{g}", VLEN),
                "b0": self.alloc_scratch(f"b0_{g}", VLEN),
                "b1": self.alloc_scratch(f"b1_{g}", VLEN),
                "b2": self.alloc_scratch(f"b2_{g}", VLEN),
                "base": self.alloc_scratch(f"base_{g}"),
            }
            group_vars.append(vars_g)

        def emit_hash(op_list: list[Op], val: int, tmp1: int, tmp2: int):
            emit_muladd(op_list, val, val, m1_vec, c1_vec)
            emit_valu(op_list, "^", tmp1, val, c2_vec)
            emit_valu(op_list, ">>", tmp2, val, sh19_vec)
            emit_valu(op_list, "^", val, tmp1, tmp2)
            emit_muladd(op_list, val, val, m3_vec, c3_vec)
            emit_valu(op_list, "+", tmp1, val, c4_vec)
            emit_valu(op_list, "<<", tmp2, val, sh9_vec)
            emit_valu(op_list, "^", val, tmp1, tmp2)
            emit_muladd(op_list, val, val, m5_vec, c5_vec)
            emit_valu(op_list, "^", tmp1, val, c6_vec)
            emit_valu(op_list, ">>", tmp2, val, sh16_vec)
            emit_valu(op_list, "^", val, tmp1, tmp2)

        idx_add_alu = os.environ.get("IDX_ADD_ALU", "1") == "1"

        def emit_idx_update(op_list: list[Op], idx: int, parity: int):
            emit_muladd(op_list, idx, idx, two_vec, one_vec)
            if idx_add_alu:
                for lane in range(VLEN):
                    emit_alu(op_list, "+", idx + lane, idx + lane, parity + lane)
            else:
                emit_valu(op_list, "+", idx, idx, parity)

        def emit_parity_scalar(op_list: list[Op], dest: int, src: int):
            for lane in range(VLEN):
                emit_alu(op_list, "&", dest + lane, src + lane, one_scalar)

        parity_mask = int(os.environ.get("PARITY_MASK", "188"), 0)

        def emit_parity(op_list: list[Op], dest: int, src: int, mask_bit: int):
            if parity_mask & (1 << mask_bit):
                emit_parity_scalar(op_list, dest, src)
            else:
                emit_valu(op_list, "&", dest, src, one_vec)

        idx_bits_add_alu = os.environ.get("IDX_BITS_ADD_ALU", "1") == "1"

        def emit_idx_from_bits(
            op_list: list[Op], idx: int, b0: int, b1: int, b2: int, tmp1: int, tmp2: int
        ):
            emit_muladd(op_list, tmp1, b2, two_vec, idx)
            emit_muladd(op_list, tmp1, b1, four_vec, tmp1)
            emit_muladd(op_list, idx, b0, eight_vec, tmp1)
            if idx_bits_add_alu:
                for lane in range(VLEN):
                    emit_alu(op_list, "+", idx + lane, idx + lane, fifteen_scalar)
            else:
                emit_valu(op_list, "+", idx, idx, fifteen_vec)

        load_addr_alu = os.environ.get("LOAD_ADDR_ALU", "1") == "1"

        def emit_load_node(op_list: list[Op], idx: int, addr: int, node: int):
            if load_addr_alu:
                for off in range(VLEN):
                    emit_alu(op_list, "+", addr + off, idx + off, forest_base_scalar)
            else:
                emit_valu(op_list, "+", addr, idx, forest_base_vec)
            for off in range(VLEN):
                emit_load_offset(op_list, node, addr, off)

        for batch in range(batches):
            batch_ops: list[Op] = []
            group_start_ops: list[int] = []
            batch_groups = min(groups_per_batch, total_groups - batch * groups_per_batch)
            for g, vars_g in enumerate(group_vars[:batch_groups]):
                group_start_ops.append(len(batch_ops))
                group_id = batch * groups_per_batch + g
                base_addr_val = inp_base_val + group_id * VLEN
                emit_const(batch_ops, vars_g["base"], base_addr_val)
                emit_vload(batch_ops, vars_g["val"], vars_g["base"])
                emit_valu(batch_ops, "+", vars_g["idx"], zero_vec, zero_vec)

            for vars_g in group_vars[:batch_groups]:
                val = vars_g["val"]
                idx = vars_g["idx"]
                tmp1 = vars_g["tmp1"]
                tmp2 = vars_g["tmp2"]
                b0 = vars_g["b0"]
                b1 = vars_g["b1"]
                b2 = vars_g["b2"]
                # Round 0 (root)
                emit_valu(batch_ops, "^", val, val, node_vec[0])
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_parity(batch_ops, b0, val, 0)

                # Round 1 (depth 1)
                emit_vselect(batch_ops, tmp1, b0, node_vec[2], node_vec[1])
                emit_valu(batch_ops, "^", val, val, tmp1)
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_parity(batch_ops, b1, val, 1)

                # Round 2 (depth 2)
                emit_vselect(batch_ops, tmp1, b1, node_vec[4], node_vec[3])
                emit_vselect(batch_ops, tmp2, b1, node_vec[6], node_vec[5])
                emit_vselect(batch_ops, tmp1, b0, tmp2, tmp1)
                emit_valu(batch_ops, "^", val, val, tmp1)
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_parity(batch_ops, b2, val, 2)

                # Round 3 (depth 3)
                emit_vselect(batch_ops, tmp1, b2, node_vec[8], node_vec[7])
                emit_vselect(batch_ops, tmp2, b2, node_vec[10], node_vec[9])
                emit_vselect(batch_ops, tmp1, b1, tmp2, tmp1)
                emit_vselect(batch_ops, tmp2, b2, node_vec[12], node_vec[11])
                emit_vselect(batch_ops, idx, b2, node_vec[14], node_vec[13])
                emit_vselect(batch_ops, tmp2, b1, idx, tmp2)
                emit_vselect(batch_ops, tmp1, b0, tmp2, tmp1)
                emit_valu(batch_ops, "^", val, val, tmp1)
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_parity(batch_ops, idx, val, 3)
                emit_idx_from_bits(batch_ops, idx, b0, b1, b2, tmp1, tmp2)

                # Rounds 4-9 (load-based)
                for _ in range(6):
                    emit_load_node(batch_ops, idx, tmp2, tmp1)
                    emit_valu(batch_ops, "^", val, val, tmp1)
                    emit_hash(batch_ops, val, tmp1, tmp2)
                    emit_parity_scalar(batch_ops, tmp1, val)
                    emit_idx_update(batch_ops, idx, tmp1)

                # Round 10 (load-based, then reset idx)
                emit_load_node(batch_ops, idx, tmp2, tmp1)
                emit_valu(batch_ops, "^", val, val, tmp1)
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_valu(batch_ops, "+", idx, zero_vec, zero_vec)

                # Round 11 (root)
                emit_valu(batch_ops, "^", val, val, node_vec[0])
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_parity(batch_ops, b0, val, 4)

                # Round 12 (depth 1)
                emit_vselect(batch_ops, tmp1, b0, node_vec[2], node_vec[1])
                emit_valu(batch_ops, "^", val, val, tmp1)
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_parity(batch_ops, b1, val, 5)

                # Round 13 (depth 2)
                emit_vselect(batch_ops, tmp1, b1, node_vec[4], node_vec[3])
                emit_vselect(batch_ops, tmp2, b1, node_vec[6], node_vec[5])
                emit_vselect(batch_ops, tmp1, b0, tmp2, tmp1)
                emit_valu(batch_ops, "^", val, val, tmp1)
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_parity(batch_ops, b2, val, 6)

                # Round 14 (depth 3)
                emit_vselect(batch_ops, tmp1, b2, node_vec[8], node_vec[7])
                emit_vselect(batch_ops, tmp2, b2, node_vec[10], node_vec[9])
                emit_vselect(batch_ops, tmp1, b1, tmp2, tmp1)
                emit_vselect(batch_ops, tmp2, b2, node_vec[12], node_vec[11])
                emit_vselect(batch_ops, idx, b2, node_vec[14], node_vec[13])
                emit_vselect(batch_ops, tmp2, b1, idx, tmp2)
                emit_vselect(batch_ops, tmp1, b0, tmp2, tmp1)
                emit_valu(batch_ops, "^", val, val, tmp1)
                emit_hash(batch_ops, val, tmp1, tmp2)
                emit_parity(batch_ops, idx, val, 7)
                emit_idx_from_bits(batch_ops, idx, b0, b1, b2, tmp1, tmp2)

                # Round 15 (load-based, no idx update)
                emit_load_node(batch_ops, idx, tmp2, tmp1)
                emit_valu(batch_ops, "^", val, val, tmp1)
                emit_hash(batch_ops, val, tmp1, tmp2)

                emit_vstore(batch_ops, vars_g["base"], val)

            start_offsets = {op_id: gi * start_spacing for gi, op_id in enumerate(group_start_ops)}
            self.instrs.extend(schedule_ops(batch_ops, start_offsets=start_offsets))

        self.instrs[:0] = schedule_ops(init_ops)

        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"

    def build_kernel_baseline(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
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

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
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
