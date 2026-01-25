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
import argparse
import os
import sys
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
    myhash,
)
from vliw_scheduler import schedule_slots


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}
        self.vec_broadcast_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        if _env_flag("VLIW"):
            vliw = True
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        return schedule_slots(slots, SLOT_LIMITS)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def tag(self, text: str):
        # Debug tags are ignored by submission simulator; useful for profiling.
        self.instrs.append({"debug": [("tag", text)]})

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

    def scratch_vconst(self, val, name=None):
        if val not in self.vec_const_map:
            scalar = self.scratch_const(val, name=None)
            addr = self.alloc_scratch(name, length=VLEN)
            self.add("valu", ("vbroadcast", addr, scalar))
            self.vec_const_map[val] = addr
        return self.vec_const_map[val]

    def scratch_vbroadcast(self, scalar_addr, name=None):
        if scalar_addr not in self.vec_broadcast_map:
            addr = self.alloc_scratch(name, length=VLEN)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vec_broadcast_map[scalar_addr] = addr
        return self.vec_broadcast_map[scalar_addr]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vec(self, val_vec, tmp1_vec, tmp2_vec, round, base):
        return self.build_hash_vec_multi([(val_vec, tmp1_vec, tmp2_vec)], round, base)

    def build_hash_vec_multi(self, val_groups, round, base):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op2 == "+" and op1 == "+" and op3 == "<<":
                # (a + val1) + (a << k) == a * (1 + 2^k) + val1
                mul = self.scratch_vconst(1 + (1 << val3))
                add = self.scratch_vconst(val1)
                for val_vec, _t1, _t2 in val_groups:
                    slots.append(("valu", ("multiply_add", val_vec, val_vec, mul, add)))
                continue
            c1 = self.scratch_vconst(val1)
            c3 = self.scratch_vconst(val3)
            for val_vec, t1, _t2 in val_groups:
                slots.append(("valu", (op1, t1, val_vec, c1)))
            for val_vec, _t1, t2 in val_groups:
                slots.append(("valu", (op3, t2, val_vec, c3)))
            if _env_flag("SCALAR_OP2") and op2 in ("+", "^"):
                for val_vec, t1, t2 in val_groups:
                    for lane in range(VLEN):
                        slots.append(("alu", (op2, val_vec + lane, t1 + lane, t2 + lane)))
            else:
                for val_vec, t1, t2 in val_groups:
                    slots.append(("valu", (op2, val_vec, t1, t2)))
        return slots

    def build_vselect_vec(self, dest, cond, a, b, tmp_mask, tmp_notmask, vec_ones, vec_zero):
        # Implement vselect via VALU ops to avoid flow bottleneck.
        # mask = 0 - cond (0 -> 0x0, 1 -> 0xFFFFFFFF)
        slots = []
        slots.append(("valu", ("-", tmp_mask, vec_zero, cond)))
        slots.append(("valu", ("^", tmp_notmask, tmp_mask, vec_ones)))
        slots.append(("valu", ("&", dest, a, tmp_mask)))
        slots.append(("valu", ("&", tmp_mask, b, tmp_notmask)))
        slots.append(("valu", ("|", dest, dest, tmp_mask)))
        return slots

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        *,
        vliw: bool = False,
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

        # Scalar scratch registers (used in both paths)
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        if _env_flag("VEC"):
            # Unroll factor for vector chunks (1, 2, 4, 8)
            if _env_flag("VEC2"):
                unroll = 2
            elif _env_flag("VEC4"):
                unroll = 4
            else:
                try:
                    unroll = int(os.getenv("VEC_UNROLL", "1"))
                except ValueError:
                    unroll = 1
            if unroll not in (1, 2, 4, 8, 12, 16):
                unroll = 1
            if unroll > 8:
                # Above 8, scratch usage and dependency complexity spike; cap for correctness.
                unroll = 8

            inp_scratch = _env_flag("INP_SCRATCH")
            vec_tmp3 = self.alloc_scratch("vec_tmp3", length=VLEN)

            # Allocate vector pools to avoid scratch overlap between different roles.
            vec_idx_base = self.alloc_scratch("vec_idx_pool", length=unroll * VLEN)
            vec_val_base = self.alloc_scratch("vec_val_pool", length=unroll * VLEN)
            vec_node_base = self.alloc_scratch("vec_node_pool", length=unroll * VLEN)
            vec_tmp1_base = self.alloc_scratch("vec_tmp1_pool", length=unroll * VLEN)
            vec_tmp2_base = self.alloc_scratch("vec_tmp2_pool", length=unroll * VLEN)
            vec_addr_base = self.alloc_scratch("vec_addr_pool", length=unroll * VLEN)
            vec_addr_idx_base = self.alloc_scratch("vec_addr_idx_pool", length=unroll)
            vec_addr_val_base = self.alloc_scratch("vec_addr_val_pool", length=unroll)

            vec_idx_u = [vec_idx_base + ui * VLEN for ui in range(unroll)]
            vec_val_u = [vec_val_base + ui * VLEN for ui in range(unroll)]
            vec_node_u = [vec_node_base + ui * VLEN for ui in range(unroll)]
            vec_tmp1_u = [vec_tmp1_base + ui * VLEN for ui in range(unroll)]
            vec_tmp2_u = [vec_tmp2_base + ui * VLEN for ui in range(unroll)]
            vec_addr_u = [vec_addr_base + ui * VLEN for ui in range(unroll)]
            vec_addr_idx_u = [vec_addr_idx_base + ui for ui in range(unroll)]
            vec_addr_val_u = [vec_addr_val_base + ui for ui in range(unroll)]
            if inp_scratch:
                idx_buf = self.alloc_scratch("idx_buf", length=batch_size)
                val_buf = self.alloc_scratch("val_buf", length=batch_size)
            vec_zero = self.scratch_vconst(0)
            vec_one = self.scratch_vconst(1)
            vec_two = self.scratch_vconst(2)
            vec_ones = self.scratch_vconst(0xFFFFFFFF)
            vec_n_nodes = self.scratch_vbroadcast(self.scratch["n_nodes"])
            vec_forest_base = self.scratch_vbroadcast(self.scratch["forest_values_p"])
            vselect_valu = _env_flag("VSELECT_VALU")
            per_value_pipe = _env_flag("PER_VALUE_PIPE")
            arith_select = _env_flag("ARITH_SELECT")
            parity_and = _env_flag("PARITY_AND")
            arith_wrap = _env_flag("ARITH_WRAP")
            skip_wrap = _env_flag("SKIP_WRAP")
            small_gather = _env_flag("SMALL_GATHER")
            if small_gather:
                forest0 = self.alloc_scratch("forest0")
                forest1 = self.alloc_scratch("forest1")
                forest2 = self.alloc_scratch("forest2")
                vec_forest0 = self.alloc_scratch("vec_forest0", length=VLEN)
                vec_forest1 = self.alloc_scratch("vec_forest1", length=VLEN)
                vec_forest2 = self.alloc_scratch("vec_forest2", length=VLEN)
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], zero_const)))
                body.append(("load", ("load", forest0, tmp_addr)))
                body.append(("valu", ("vbroadcast", vec_forest0, forest0)))
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], one_const)))
                body.append(("load", ("load", forest1, tmp_addr)))
                body.append(("valu", ("vbroadcast", vec_forest1, forest1)))
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], two_const)))
                body.append(("load", ("load", forest2, tmp_addr)))
                body.append(("valu", ("vbroadcast", vec_forest2, forest2)))

            vec_limit = (batch_size // VLEN) * VLEN
            if per_value_pipe:
                # Per-value pipeline: load each vector chunk once, run all rounds, then store once.
                for base in range(0, vec_limit, VLEN * unroll):
                    # address setup + vload idx/val
                    for ui in range(unroll):
                        base_u = base + ui * VLEN
                        if base_u >= vec_limit:
                            continue
                        base_u_const = self.scratch_const(base_u)
                        body.append(
                            (
                                "alu",
                                (
                                    "+",
                                    vec_addr_idx_u[ui],
                                    self.scratch["inp_indices_p"],
                                    base_u_const,
                                ),
                            )
                        )
                        body.append(
                            (
                                "alu",
                                (
                                    "+",
                                    vec_addr_val_u[ui],
                                    self.scratch["inp_values_p"],
                                    base_u_const,
                                ),
                            )
                        )
                        body.append(("load", ("vload", vec_idx_u[ui], vec_addr_idx_u[ui])))
                        body.append(("load", ("vload", vec_val_u[ui], vec_addr_val_u[ui])))

                    for round in range(rounds):
                        depth = round % (forest_height + 1)
                        # gather node values
                        if not (small_gather and depth in (0, 1)):
                            for ui in range(unroll):
                                base_u = base + ui * VLEN
                                if base_u >= vec_limit:
                                    continue
                                body.append(
                                    (
                                        "valu",
                                        ("+", vec_addr_u[ui], vec_forest_base, vec_idx_u[ui]),
                                    )
                                )
                                for lane in range(VLEN):
                                    body.append(
                                        (
                                            "load",
                                            ("load_offset", vec_node_u[ui], vec_addr_u[ui], lane),
                                        )
                                    )

                        # process each vector (hash interleaved across unroll)
                        val_groups = []
                        idx_vecs = []
                        for ui in range(unroll):
                            base_u = base + ui * VLEN
                            if base_u >= vec_limit:
                                continue
                            if small_gather and depth == 0:
                                body.append(
                                    (
                                        "valu",
                                        ("^", vec_val_u[ui], vec_val_u[ui], vec_forest0),
                                    )
                                )
                            elif small_gather and depth == 1:
                                # idx is 1 or 2; select forest1/forest2 based on idx % 2
                                body.append(("valu", ("%", vec_tmp1_u[ui], vec_idx_u[ui], vec_two)))
                                if arith_select:
                                    # node = forest2 ^ ((forest1 ^ forest2) & mask)
                                    body.append(
                                        ("valu", ("^", vec_tmp2_u[ui], vec_forest1, vec_forest2))
                                    )
                                    body.append(("valu", ("-", vec_tmp3, vec_zero, vec_tmp1_u[ui])))
                                    body.append(
                                        ("valu", ("&", vec_tmp2_u[ui], vec_tmp2_u[ui], vec_tmp3))
                                    )
                                    body.append(
                                        ("valu", ("^", vec_node_u[ui], vec_forest2, vec_tmp2_u[ui]))
                                    )
                                elif vselect_valu:
                                    body.extend(
                                        self.build_vselect_vec(
                                            vec_node_u[ui],
                                            vec_tmp1_u[ui],
                                            vec_forest1,
                                            vec_forest2,
                                            vec_tmp1_u[ui],
                                            vec_tmp2_u[ui],
                                            vec_ones,
                                            vec_zero,
                                        )
                                    )
                                else:
                                    body.append(
                                        ("flow", ("vselect", vec_node_u[ui], vec_tmp1_u[ui], vec_forest1, vec_forest2))
                                    )
                                body.append(
                                    (
                                        "valu",
                                        ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui]),
                                    )
                                )
                            else:
                                body.append(
                                    (
                                        "valu",
                                        ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui]),
                                    )
                                )
                            val_groups.append((vec_val_u[ui], vec_tmp1_u[ui], vec_tmp2_u[ui]))
                            idx_vecs.append(vec_idx_u[ui])

                        body.extend(self.build_hash_vec_multi(val_groups, round, base))

                        for vi, (val_vec, t1, _t2) in enumerate(val_groups):
                            idx_vec = idx_vecs[vi]
                            if parity_and:
                                body.append(("valu", ("&", t1, val_vec, vec_one)))
                                body.append(("valu", ("+", vec_tmp3, t1, vec_one)))
                            else:
                                body.append(("valu", ("%", t1, val_vec, vec_two)))
                                body.append(("valu", ("==", t1, t1, vec_zero)))
                                if arith_select:
                                    body.append(("valu", ("-", vec_tmp3, vec_two, t1)))
                                elif vselect_valu:
                                    body.extend(
                                        self.build_vselect_vec(
                                            vec_tmp3,
                                            t1,
                                            vec_one,
                                            vec_two,
                                            t1,
                                            vec_tmp2_u[vi],
                                            vec_ones,
                                            vec_zero,
                                        )
                                    )
                                else:
                                    body.append(
                                        ("flow", ("vselect", vec_tmp3, t1, vec_one, vec_two))
                                    )
                            body.append(("valu", ("*", idx_vec, idx_vec, vec_two)))
                            body.append(("valu", ("+", idx_vec, idx_vec, vec_tmp3)))
                            if not (skip_wrap and depth < forest_height):
                                body.append(("valu", ("<", t1, idx_vec, vec_n_nodes)))
                                if arith_wrap:
                                    body.append(("valu", ("*", idx_vec, idx_vec, t1)))
                                elif arith_select:
                                    body.append(("valu", ("*", idx_vec, idx_vec, t1)))
                                elif vselect_valu:
                                    body.extend(
                                        self.build_vselect_vec(
                                            idx_vec,
                                            t1,
                                            idx_vec,
                                            vec_zero,
                                            t1,
                                            vec_tmp2_u[vi],
                                            vec_ones,
                                            vec_zero,
                                        )
                                    )
                                else:
                                    body.append(
                                        ("flow", ("vselect", idx_vec, t1, idx_vec, vec_zero))
                                    )

                    # store idx/val once per chunk
                    for ui in range(unroll):
                        base_u = base + ui * VLEN
                        if base_u >= vec_limit:
                            continue
                        body.append(("store", ("vstore", vec_addr_idx_u[ui], vec_idx_u[ui])))
                        body.append(("store", ("vstore", vec_addr_val_u[ui], vec_val_u[ui])))

                # tail (scalar)
                for i in range(vec_limit, batch_size):
                    i_const = self.scratch_const(i)
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                    body.append(("load", ("load", tmp_idx, tmp_addr)))
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                    body.append(("load", ("load", tmp_val, tmp_addr)))
                    for round in range(rounds):
                        depth = round % (forest_height + 1)
                        body.append(
                            ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                        )
                        body.append(("load", ("load", tmp_node_val, tmp_addr)))
                        body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                        body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                        body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                        body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                        body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                        body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                        body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                        if not (skip_wrap and depth < forest_height):
                            body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                            body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                    body.append(("store", ("store", tmp_addr, tmp_idx)))
                    body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                    body.append(("store", ("store", tmp_addr, tmp_val)))
            else:
                for round in range(rounds):
                    depth = round % (forest_height + 1)
                    if inp_scratch and round == 0:
                        for base in range(0, vec_limit, VLEN):
                            base_const = self.scratch_const(base)
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], base_const))
                            )
                            body.append(("load", ("vload", idx_buf + base, tmp_addr)))
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], base_const))
                            )
                            body.append(("load", ("vload", val_buf + base, tmp_addr)))
                        for i in range(vec_limit, batch_size):
                            i_const = self.scratch_const(i)
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                            )
                            body.append(("load", ("load", idx_buf + i, tmp_addr)))
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                            )
                            body.append(("load", ("load", val_buf + i, tmp_addr)))
                    if unroll > 0:
                        for base in range(0, vec_limit, VLEN * unroll):
                            # address setup
                            for ui in range(unroll):
                                base_u = base + ui * VLEN
                                if base_u >= vec_limit:
                                    continue
                                if not inp_scratch:
                                    base_u_const = self.scratch_const(base_u)
                                    body.append(
                                        (
                                            "alu",
                                            (
                                                "+",
                                                vec_addr_idx_u[ui],
                                                self.scratch["inp_indices_p"],
                                                base_u_const,
                                            ),
                                        )
                                    )
                                    body.append(
                                        (
                                            "alu",
                                            (
                                                "+",
                                                vec_addr_val_u[ui],
                                                self.scratch["inp_values_p"],
                                                base_u_const,
                                            ),
                                        )
                                    )

                            # vload idx/val
                            for ui in range(unroll):
                                base_u = base + ui * VLEN
                                if base_u >= vec_limit:
                                    continue
                                if inp_scratch:
                                    vec_idx_u[ui] = idx_buf + base_u
                                    vec_val_u[ui] = val_buf + base_u
                                else:
                                    self.tag(f"round={round} base={base_u} ui={ui} kind=idx")
                                    body.append(
                                        ("load", ("vload", vec_idx_u[ui], vec_addr_idx_u[ui]))
                                    )
                                    self.tag(f"round={round} base={base_u} ui={ui} kind=val")
                                    body.append(
                                        ("load", ("vload", vec_val_u[ui], vec_addr_val_u[ui]))
                                    )

                            # gather node values
                            depth = round % (forest_height + 1)
                            if not (small_gather and depth in (0, 1)):
                                for ui in range(unroll):
                                    base_u = base + ui * VLEN
                                    if base_u >= vec_limit:
                                        continue
                                    body.append(
                                        (
                                            "valu",
                                            ("+", vec_addr_u[ui], vec_forest_base, vec_idx_u[ui]),
                                        )
                                    )
                                    for lane in range(VLEN):
                                        body.append(
                                            (
                                                "load",
                                                ("load_offset", vec_node_u[ui], vec_addr_u[ui], lane),
                                            )
                                        )

                            # process each vector (hash interleaved across unroll)
                            val_groups = []
                            idx_vecs = []
                            addr_idx_vecs = []
                            addr_val_vecs = []
                            for ui in range(unroll):
                                base_u = base + ui * VLEN
                                if base_u >= vec_limit:
                                    continue
                                if small_gather and depth == 0:
                                    body.append(
                                        (
                                            "valu",
                                            ("^", vec_val_u[ui], vec_val_u[ui], vec_forest0),
                                        )
                                    )
                                elif small_gather and depth == 1:
                                    # idx is 1 or 2; select forest1/forest2 based on idx % 2
                                    body.append(("valu", ("%", vec_tmp1_u[ui], vec_idx_u[ui], vec_two)))
                                    if arith_select:
                                        # node = forest2 ^ ((forest1 ^ forest2) & mask)
                                        body.append(
                                            ("valu", ("^", vec_tmp2_u[ui], vec_forest1, vec_forest2))
                                        )
                                        body.append(("valu", ("-", vec_tmp3, vec_zero, vec_tmp1_u[ui])))
                                        body.append(
                                            ("valu", ("&", vec_tmp2_u[ui], vec_tmp2_u[ui], vec_tmp3))
                                        )
                                        body.append(
                                            ("valu", ("^", vec_node_u[ui], vec_forest2, vec_tmp2_u[ui]))
                                        )
                                    elif vselect_valu:
                                        body.extend(
                                            self.build_vselect_vec(
                                                vec_node_u[ui],
                                                vec_tmp1_u[ui],
                                                vec_forest1,
                                                vec_forest2,
                                                vec_tmp1_u[ui],
                                                vec_tmp2_u[ui],
                                                vec_ones,
                                                vec_zero,
                                            )
                                        )
                                    else:
                                        body.append(
                                            ("flow", ("vselect", vec_node_u[ui], vec_tmp1_u[ui], vec_forest1, vec_forest2))
                                        )
                                    body.append(
                                        (
                                            "valu",
                                            ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui]),
                                        )
                                    )
                                else:
                                    body.append(
                                        (
                                            "valu",
                                            ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui]),
                                        )
                                    )
                                val_groups.append((vec_val_u[ui], vec_tmp1_u[ui], vec_tmp2_u[ui]))
                                idx_vecs.append(vec_idx_u[ui])
                                addr_idx_vecs.append(vec_addr_idx_u[ui])
                                addr_val_vecs.append(vec_addr_val_u[ui])

                            body.extend(
                                self.build_hash_vec_multi(
                                    val_groups, round, base
                                )
                            )

                            for vi, (val_vec, t1, _t2) in enumerate(val_groups):
                                idx_vec = idx_vecs[vi]
                                if parity_and:
                                    body.append(("valu", ("&", t1, val_vec, vec_one)))
                                    body.append(("valu", ("+", vec_tmp3, t1, vec_one)))
                                else:
                                    body.append(("valu", ("%", t1, val_vec, vec_two)))
                                    body.append(("valu", ("==", t1, t1, vec_zero)))
                                    if arith_select:
                                        body.append(("valu", ("-", vec_tmp3, vec_two, t1)))
                                    elif vselect_valu:
                                        body.extend(
                                            self.build_vselect_vec(
                                                vec_tmp3,
                                                t1,
                                                vec_one,
                                                vec_two,
                                                t1,
                                                vec_tmp2_u[vi],
                                                vec_ones,
                                                vec_zero,
                                            )
                                        )
                                    else:
                                        body.append(
                                            ("flow", ("vselect", vec_tmp3, t1, vec_one, vec_two))
                                        )
                                body.append(("valu", ("*", idx_vec, idx_vec, vec_two)))
                                body.append(("valu", ("+", idx_vec, idx_vec, vec_tmp3)))
                                if not (skip_wrap and depth < forest_height):
                                    body.append(("valu", ("<", t1, idx_vec, vec_n_nodes)))
                                    if arith_wrap:
                                        body.append(("valu", ("*", idx_vec, idx_vec, t1)))
                                    elif arith_select:
                                        body.append(("valu", ("*", idx_vec, idx_vec, t1)))
                                    elif vselect_valu:
                                        body.extend(
                                            self.build_vselect_vec(
                                                idx_vec,
                                                t1,
                                                idx_vec,
                                                vec_zero,
                                                t1,
                                                vec_tmp2_u[vi],
                                                vec_ones,
                                                vec_zero,
                                            )
                                        )
                                    else:
                                        body.append(
                                            ("flow", ("vselect", idx_vec, t1, idx_vec, vec_zero))
                                        )
                                if not inp_scratch:
                                    body.append(
                                        ("store", ("vstore", addr_idx_vecs[vi], idx_vec))
                                    )
                                    body.append(
                                        ("store", ("vstore", addr_val_vecs[vi], val_vec))
                                    )
                    # tail (scalar)
                    for i in range(vec_limit, batch_size):
                        if inp_scratch:
                            idx_addr = idx_buf + i
                            val_addr = val_buf + i
                            body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_addr)))
                            body.append(("load", ("load", tmp_node_val, tmp_addr)))
                            body.append(("alu", ("^", val_addr, val_addr, tmp_node_val)))
                            body.extend(self.build_hash(val_addr, tmp1, tmp2, round, i))
                            body.append(("alu", ("%", tmp1, val_addr, two_const)))
                            body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                            body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                            body.append(("alu", ("*", idx_addr, idx_addr, two_const)))
                            body.append(("alu", ("+", idx_addr, idx_addr, tmp3)))
                            body.append(("alu", ("<", tmp1, idx_addr, self.scratch["n_nodes"])))
                            body.append(("flow", ("select", idx_addr, tmp1, idx_addr, zero_const)))
                        else:
                            i_const = self.scratch_const(i)
                            body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                            body.append(("load", ("load", tmp_idx, tmp_addr)))
                            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                            body.append(("load", ("load", tmp_val, tmp_addr)))
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                            )
                            body.append(("load", ("load", tmp_node_val, tmp_addr)))
                        body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                        body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                        body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                        body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                        body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                        body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                        body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                        if not (skip_wrap and depth < forest_height):
                            body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                            body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                            body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                            body.append(("store", ("store", tmp_addr, tmp_idx)))
                            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                            body.append(("store", ("store", tmp_addr, tmp_val)))
                    if inp_scratch and round == rounds - 1:
                        for base in range(0, vec_limit, VLEN):
                            base_const = self.scratch_const(base)
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], base_const))
                            )
                            body.append(("store", ("vstore", tmp_addr, idx_buf + base)))
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], base_const))
                            )
                            body.append(("store", ("vstore", tmp_addr, val_buf + base)))
                        for i in range(vec_limit, batch_size):
                            i_const = self.scratch_const(i)
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                            )
                            body.append(("store", ("store", tmp_addr, idx_buf + i)))
                            body.append(
                                ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                            )
                            body.append(("store", ("store", tmp_addr, val_buf + i)))
        else:
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
                    body.append(
                        ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                    )
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

        body_instrs = self.build(body, vliw=vliw)
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
    vliw: bool = False,
):
    if _env_flag("BITSLICE_PROFILE"):
        _bitslice_profile()
    if _env_flag("IDX_PROFILE"):
        random.seed(seed)
        forest = Tree.generate(forest_height)
        inp = Input.generate(forest, batch_size, rounds)
        total = 0
        repeats = 0
        per_round = []
        for h in range(rounds):
            seen = set()
            for idx in inp.indices:
                total += 1
                if idx in seen:
                    repeats += 1
                else:
                    seen.add(idx)
            per_round.append((len(seen), batch_size - len(seen)))
            for i in range(len(inp.indices)):
                idx = inp.indices[i]
                val = inp.values[i]
                val = myhash(val ^ forest.values[idx])
                idx = 2 * idx + (1 if val % 2 == 0 else 2)
                idx = 0 if idx >= len(forest.values) else idx
                inp.values[i] = val
                inp.indices[i] = idx
        print(f"IDX_PROFILE: total={total} repeats={repeats} hit_rate={repeats/total:.4f}")
        for r, (uniq, rep) in enumerate(per_round):
            print(f"  round {r}: unique={uniq} repeats={rep}")
    if _env_flag("VLIW"):
        vliw = True
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(
        forest.height,
        len(forest.values),
        len(inp.indices),
        rounds,
        vliw=vliw,
    )
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
    if _env_flag("PROFILE"):
        machine.profile = True
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


def _bitslice_profile(vlen: int = VLEN):
    """
    Prototype: estimate bit-sliced op cost for the hash.
    This does NOT change kernel behavior; it only prints an estimate.
    """
    # Bit-sliced add: ripple-carry using xor/and/or
    def add_cost(bits: int):
        # sum = a ^ b ^ c; carry = (a&b) | (c & (a^b))
        # per bit: xor*2, and*2, or*1 => 5 ops
        return bits * 5

    # Bit-sliced xor cost: 1 op per bit
    def xor_cost(bits: int):
        return bits

    # Bit-sliced shift cost: treat as wiring (no ops), but in software you'd just reindex.
    def shift_cost(_bits: int):
        return 0

    total = 0
    bits = 32
    for op1, _val1, op2, op3, _val3 in HASH_STAGES:
        # op1(a, const) cost
        total += add_cost(bits) if op1 == "+" else xor_cost(bits)
        # op3(a, shift) cost
        total += shift_cost(bits)
        # op2(t1, t2) cost
        if op2 == "+":
            total += add_cost(bits)
        elif op2 == "^":
            total += xor_cost(bits)
    print(f"BITSLICE_PROFILE: est bitwise ops per value = {total}")
    print("  note: ignores data movement + gather; intended as rough feasibility check")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run kernel test.")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--prints", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline and VLIW back-to-back with the same parameters.",
    )
    return parser.parse_args()


def _env_flag(name: str) -> bool:
    val = os.getenv(name, "").strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


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


def _main():
    args = _parse_args()
    vliw = _env_flag("VLIW")
    if args.compare:
        print("== Baseline ==")
        do_kernel_test(
            args.forest_height,
            args.rounds,
            args.batch_size,
            seed=args.seed,
            trace=args.trace,
            prints=args.prints,
            vliw=False,
        )
        print("== VLIW ==")
        do_kernel_test(
            args.forest_height,
            args.rounds,
            args.batch_size,
            seed=args.seed,
            trace=args.trace,
            prints=args.prints,
            vliw=True,
        )
    else:
        do_kernel_test(
            args.forest_height,
            args.rounds,
            args.batch_size,
            seed=args.seed,
            trace=args.trace,
            prints=args.prints,
            vliw=vliw,
        )


def _has_cli_args(argv: list[str]) -> bool:
    cli_flags = {
        "--forest-height",
        "--rounds",
        "--batch-size",
        "--seed",
        "--trace",
        "--prints",
        "--compare",
    }
    return any(arg in cli_flags for arg in argv[1:])


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
    if _has_cli_args(sys.argv):
        _main()
    else:
        unittest.main()
