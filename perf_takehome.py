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
        self.drop_valu_safe_ranges = []

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]]):
        # Always use VLIW scheduling (best setting).
        drop_valu = int(os.getenv("DROP_VALU", "0") or 0)
        drop_safe = os.getenv("DROP_VALU_SAFE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if drop_valu > 0:
            rng = random.Random(0)
            if drop_safe and self.drop_valu_safe_ranges:
                valu_idxs = [
                    i
                    for i, (engine, slot) in enumerate(slots)
                    if engine == "valu" and self._drop_valu_safe_slot(slot)
                ]
            else:
                valu_idxs = [
                    i for i, (engine, _slot) in enumerate(slots) if engine == "valu"
                ]
            if drop_valu > len(valu_idxs):
                drop_valu = len(valu_idxs)
            drop_set = set(rng.sample(valu_idxs, drop_valu))
            slots = [slot for i, slot in enumerate(slots) if i not in drop_set]
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

    def mark_drop_valu_safe(self, start, length):
        self.drop_valu_safe_ranges.append((start, start + length))

    def _drop_valu_safe_slot(self, slot):
        # Only drop VALU ops that write into explicitly safe vector ranges.
        dest = slot[1]
        for start, end in self.drop_valu_safe_ranges:
            if start <= dest and dest + VLEN <= end:
                return True
        return False

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
        # Best-settings: op1/op3 stay vectorized, op2 uses scalar ALU per lane.
        scalar_op1 = False
        scalar_op3 = False
        scalar_op2 = True
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
            if scalar_op1:
                for val_vec, t1, _t2 in val_groups:
                    for lane in range(VLEN):
                        slots.append(("alu", (op1, t1 + lane, val_vec + lane, c1 + lane)))
            else:
                for val_vec, t1, _t2 in val_groups:
                    slots.append(("valu", (op1, t1, val_vec, c1)))
            if scalar_op3:
                for val_vec, _t1, t2 in val_groups:
                    for lane in range(VLEN):
                        slots.append(("alu", (op3, t2 + lane, val_vec + lane, c3 + lane)))
            else:
                for val_vec, _t1, t2 in val_groups:
                    slots.append(("valu", (op3, t2, val_vec, c3)))
            if scalar_op2 and op2 in ("+", "^"):
                for val_vec, t1, t2 in val_groups:
                    for lane in range(VLEN):
                        slots.append(("alu", (op2, val_vec + lane, t1 + lane, t2 + lane)))
            else:
                for val_vec, t1, t2 in val_groups:
                    slots.append(("valu", (op2, val_vec, t1, t2)))
        return slots

    def build_small_gather_select4_flow(
        self,
        dest,
        idx_vec,
        base_vec,
        vals4,
        tmp_idx,
        tmp_bit0,
        tmp_pair,
        vec_one,
    ):
        # 4-way select using 3 flow vselects; tmp_idx reused for bit1.
        slots = []
        slots.append(("valu", ("-", tmp_idx, idx_vec, base_vec)))
        slots.append(("valu", ("&", tmp_bit0, tmp_idx, vec_one)))
        slots.append(("valu", (">>", tmp_idx, tmp_idx, vec_one)))
        slots.append(("valu", ("&", tmp_idx, tmp_idx, vec_one)))
        slots.append(("flow", ("vselect", dest, tmp_bit0, vals4[1], vals4[0])))
        slots.append(("flow", ("vselect", tmp_pair, tmp_bit0, vals4[3], vals4[2])))
        slots.append(("flow", ("vselect", dest, tmp_idx, tmp_pair, dest)))
        return slots


    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
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
        vlen_const = self.scratch_const(VLEN)

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

        # Best-settings vector path.
        unroll = 20
        vec_tmp3 = self.alloc_scratch("vec_tmp3", length=VLEN)
        vec_tmp4 = self.alloc_scratch("vec_tmp4", length=VLEN)

        # Allocate vector pools to avoid scratch overlap between different roles.
        # vec_addr_pool holds per-lane node addresses (forest base + idx).
        vec_addr_base = self.alloc_scratch("vec_addr_pool", length=unroll * VLEN)
        vec_val_base = self.alloc_scratch("vec_val_pool", length=unroll * VLEN)
        self.mark_drop_valu_safe(vec_val_base, unroll * VLEN)
        vec_node_base = self.alloc_scratch("vec_node_pool", length=unroll * VLEN)
        vec_tmp1_base = self.alloc_scratch("vec_tmp1_pool", length=unroll * VLEN)
        vec_tmp2_base = self.alloc_scratch("vec_tmp2_pool", length=unroll * VLEN)
        vec_addr_val_base = self.alloc_scratch("vec_addr_val_pool", length=unroll)

        vec_addr_u = [vec_addr_base + ui * VLEN for ui in range(unroll)]
        vec_val_u = [vec_val_base + ui * VLEN for ui in range(unroll)]
        vec_node_u = [vec_node_base + ui * VLEN for ui in range(unroll)]
        vec_tmp1_u = [vec_tmp1_base + ui * VLEN for ui in range(unroll)]
        vec_tmp2_u = [vec_tmp2_base + ui * VLEN for ui in range(unroll)]
        vec_addr_val_u = [vec_addr_val_base + ui for ui in range(unroll)]
        vec_one = self.scratch_vconst(1)
        vec_two = self.scratch_vconst(2)
        vec_forest_base = self.scratch_vbroadcast(self.scratch["forest_values_p"])
        addr_bias = self.alloc_scratch("addr_bias")
        base_plus1 = self.alloc_scratch("base_plus1")
        vec_addr_bias = self.alloc_scratch("vec_addr_bias", length=VLEN)
        # Fixed best-settings flags.
        forest0 = self.alloc_scratch("forest0")
        forest1 = self.alloc_scratch("forest1")
        forest2 = self.alloc_scratch("forest2")
        vec_forest0 = self.alloc_scratch("vec_forest0", length=VLEN)
        vec_forest1 = self.alloc_scratch("vec_forest1", length=VLEN)
        vec_forest2 = self.alloc_scratch("vec_forest2", length=VLEN)
        vec_base1_addr = self.alloc_scratch("vec_base1_addr", length=VLEN)
        vec_base3_addr = self.alloc_scratch("vec_base3_addr", length=VLEN)
        body.append(("alu", ("-", addr_bias, one_const, self.scratch["forest_values_p"])))
        body.append(("alu", ("+", base_plus1, self.scratch["forest_values_p"], one_const)))
        body.append(("valu", ("vbroadcast", vec_addr_bias, addr_bias)))
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], zero_const)))
        body.append(("load", ("load", forest0, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest0, forest0)))
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], one_const)))
        body.append(("load", ("load", forest1, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest1, forest1)))
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], two_const)))
        body.append(("load", ("load", forest2, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest2, forest2)))
        vec_base3 = self.scratch_vconst(3)
        body.append(("valu", ("+", vec_base1_addr, vec_forest_base, vec_one)))
        body.append(("valu", ("+", vec_base3_addr, vec_forest_base, vec_base3)))
        forest3_6 = []
        forest3_6_consts = []
        vec_forest3_6 = []
        for idx in range(3, 7):
            f = self.alloc_scratch(f"forest{idx}")
            vf = self.alloc_scratch(f"vec_forest{idx}", length=VLEN)
            idx_const = self.scratch_const(idx)
            body.append(
                ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_const))
            )
            body.append(("load", ("load", f, tmp_addr)))
            body.append(("valu", ("vbroadcast", vf, f)))
            forest3_6.append(f)
            forest3_6_consts.append(idx_const)
            vec_forest3_6.append(vf)

        vec_limit = (batch_size // VLEN) * VLEN
        # Per-value pipeline: load each vector chunk once, run all rounds, then store once.
        for base in range(0, vec_limit, VLEN * unroll):
            valid_uis = []
            for ui in range(unroll):
                base_u = base + ui * VLEN
                if base_u >= vec_limit:
                    break
                valid_uis.append(ui)
            # address setup + vload values (pointer bump by VLEN)
            base_const = self.scratch_const(base)
            body.append(
                (
                    "alu",
                    ("+", vec_addr_val_u[0], self.scratch["inp_values_p"], base_const),
                )
            )
            for ui in valid_uis[1:]:
                body.append(
                    ("alu", ("+", vec_addr_val_u[ui], vec_addr_val_u[ui - 1], vlen_const))
                )
            for ui in valid_uis:
                body.append(("load", ("vload", vec_val_u[ui], vec_addr_val_u[ui])))
                body.append(
                    ("valu", ("vbroadcast", vec_addr_u[ui], self.scratch["forest_values_p"]))
                )
            for round in range(rounds):
                depth = round % (forest_height + 1)
                # gather node values
                if depth not in (0, 1, 2):
                    for ui in valid_uis:
                        for lane in range(VLEN):
                            body.append(("load", ("load_offset", vec_node_u[ui], vec_addr_u[ui], lane)))
                # process each vector (hash interleaved across unroll)
                val_groups = []
                addr_vecs = []
                for ui in valid_uis:
                    if depth == 0:
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_forest0)))
                    elif depth == 1:
                        body.append(("valu", ("-", vec_tmp1_u[ui], vec_addr_u[ui], vec_base1_addr)))
                        body.append(("flow", ("vselect", vec_node_u[ui], vec_tmp1_u[ui], vec_forest2, vec_forest1)))
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))
                    elif depth == 2:
                        body.extend(
                            self.build_small_gather_select4_flow(
                                vec_node_u[ui],
                                vec_addr_u[ui],
                                vec_base3_addr,
                                vec_forest3_6,
                                vec_tmp1_u[ui],
                                vec_tmp2_u[ui],
                                vec_tmp4,
                                vec_one,
                            )
                        )
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))
                    else:
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))
                    val_groups.append((vec_val_u[ui], vec_tmp1_u[ui], vec_tmp2_u[ui]))
                    addr_vecs.append(vec_addr_u[ui])
                body.extend(self.build_hash_vec_multi(val_groups, round, base))
                for vi, (val_vec, t1, _t2) in enumerate(val_groups):
                    addr_vec = addr_vecs[vi]
                    if depth == forest_height:
                        body.append(
                            (
                                "valu",
                                ("vbroadcast", addr_vec, self.scratch["forest_values_p"]),
                            )
                        )
                        continue
                    body.append(("valu", ("&", t1, val_vec, vec_one)))
                    if depth == 0:
                        body.append(("valu", ("+", addr_vec, vec_base1_addr, t1)))
                        continue
                    body.append(("valu", ("multiply_add", addr_vec, addr_vec, vec_two, vec_addr_bias)))
                    body.append(("valu", ("+", addr_vec, addr_vec, t1)))
                if round == rounds - 1:
                    continue
            # store values once per chunk
            for ui in valid_uis:
                body.append(("store", ("vstore", vec_addr_val_u[ui], vec_val_u[ui])))
        # tail (scalar)
        for i in range(vec_limit, batch_size):
            i_const = self.scratch_const(i)
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            body.append(("load", ("load", tmp_val, tmp_addr)))
            body.append(("alu", ("+", tmp_idx, self.scratch["forest_values_p"], zero_const)))
            for round in range(rounds):
                depth = round % (forest_height + 1)
                body.append(("load", ("load", tmp_node_val, tmp_idx)))
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                if round == rounds - 1:
                    continue
                if depth == forest_height:
                    body.append(("alu", ("+", tmp_idx, self.scratch["forest_values_p"], zero_const)))
                    continue
                body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                if depth == 0:
                    body.append(("alu", ("+", tmp_idx, base_plus1, tmp1)))
                    continue
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, addr_bias)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp1)))
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
    kb.build_kernel(
        forest.height,
        len(forest.values),
        len(inp.indices),
        rounds,
    )
    # print(kb.instrs)
    if os.getenv("ANALYZE_PATTERNS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        analyze_patterns(kb.instrs, kb.debug_info())

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
    if os.getenv("PROFILE", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
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


def analyze_patterns(instrs, debug_info=None):
    # Skip init bundles until first vload, then analyze VALU/LOAD patterns.
    start = 0
    for i, instr in enumerate(instrs):
        loads = instr.get("load", [])
        if any(slot[0] == "vload" for slot in loads):
            start = i
            break

    def _count_patterns(engine, size_filter=None):
        counts = defaultdict(int)
        for instr in instrs[start:]:
            slots = instr.get(engine, [])
            if not slots:
                continue
            if size_filter is not None and len(slots) not in size_filter:
                continue
            key = tuple(slot[0] for slot in slots)
            counts[key] += 1
        return counts

    valu_2_3 = _count_patterns("valu", size_filter={2, 3})
    load_all = _count_patterns("load", size_filter=None)

    print("=== Pattern Analysis (starting at first vload) ===")
    print(f"Start bundle index: {start}")
    if valu_2_3:
        print("VALU bundles with 2 or 3 slots (pattern -> count):")
        for pattern, count in sorted(
            valu_2_3.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"  {pattern}: {count}")
    else:
        print("No VALU bundles with 2 or 3 slots found.")

    if load_all:
        print("LOAD bundle patterns (pattern -> count):")
        for pattern, count in sorted(
            load_all.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"  {pattern}: {count}")
    else:
        print("No LOAD bundles found.")

    # Cross-tab VALU slot counts vs other engines to see if other engines are busy.
    valu_bundles = 0
    valu_slot_hist = defaultdict(int)
    cross = defaultdict(int)  # (valu_slots, alu_slots, load_slots, store_slots, flow_slots) -> count
    for instr in instrs[start:]:
        vslots = len(instr.get("valu", []))
        if vslots == 0:
            continue
        aslots = len(instr.get("alu", []))
        lslots = len(instr.get("load", []))
        sslots = len(instr.get("store", []))
        fslots = len(instr.get("flow", []))
        valu_bundles += 1
        valu_slot_hist[vslots] += 1
        cross[(vslots, aslots, lslots, sslots, fslots)] += 1

    print("VALU slot count histogram (slots -> bundles):")
    for vslots, count in sorted(valu_slot_hist.items()):
        print(f"  {vslots}: {count}")
    if valu_bundles:
        under = sum(valu_slot_hist[s] for s in (1, 2, 3))
        print(
            f"VALU underused (1-3 slots): {under}/{valu_bundles} "
            f"({under * 100.0 / valu_bundles:.1f}%)"
        )

    print("Top VALU bundles by engine occupancy (v,a,l,s,f -> count):")
    for key, count in sorted(cross.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        print(f"  {key}: {count}")

    # Phase tagging for underused VALU bundles.
    scratch_map = debug_info.scratch_map if debug_info is not None else {}

    def _range_for(name):
        for addr, (n, length) in scratch_map.items():
            if n == name:
                return (addr, addr + length)
        return None

    def _ranges_with_prefix(prefix):
        res = []
        for addr, (n, length) in scratch_map.items():
            if n.startswith(prefix):
                res.append((addr, addr + length))
        return res

    def _in_range(addr, rng):
        return rng is not None and rng[0] <= addr < rng[1]

    def _in_any(addr, ranges):
        return any(_in_range(addr, r) for r in ranges)

    val_range = _range_for("vec_val_pool")
    addr_range = _range_for("vec_addr_pool")
    tmp1_range = _range_for("vec_tmp1_pool")
    tmp2_range = _range_for("vec_tmp2_pool")
    node_range = _range_for("vec_node_pool")
    tmp4_range = _range_for("vec_tmp4")
    base1_range = _range_for("vec_base1_addr")
    base3_range = _range_for("vec_base3_addr")
    forest_ranges = _ranges_with_prefix("vec_forest")
    forest0_range = _range_for("vec_forest0")
    forest1_range = _range_for("vec_forest1")
    forest2_range = _range_for("vec_forest2")
    forest3_range = _range_for("vec_forest3")
    forest4_range = _range_for("vec_forest4")
    forest5_range = _range_for("vec_forest5")
    forest6_range = _range_for("vec_forest6")
    forest_values_p_range = _range_for("forest_values_p")

    hash_ops = {"+", "^", "<<", ">>", "multiply_add"}

    def _slot_addrs(engine, slot):
        op = slot[0]
        dest = None
        srcs = []
        if engine == "alu":
            if op in ("~",):
                dest = slot[1]
                srcs = [slot[2]]
            else:
                dest = slot[1]
                srcs = [slot[2], slot[3]]
        elif engine == "valu":
            if op == "vbroadcast":
                dest = slot[1]
                srcs = [slot[2]]
            elif op == "multiply_add":
                dest = slot[1]
                srcs = [slot[2], slot[3], slot[4]]
            else:
                dest = slot[1]
                srcs = [slot[2], slot[3]]
        elif engine == "load":
            if op in ("load", "load_offset", "vload", "const", "vbroadcast"):
                dest = slot[1]
            if op in ("load", "load_offset", "vload"):
                srcs = [slot[2]]
        elif engine == "flow":
            if op in ("select", "vselect"):
                dest = slot[1]
                srcs = [slot[2], slot[3], slot[4]]
            elif op == "add_imm":
                dest = slot[1]
                srcs = [slot[2]]
        return op, dest, srcs

    def _bundle_tags(instr):
        tags = set()
        # Gather bundles (load_offset into vec_node_pool).
        for slot in instr.get("load", []):
            if slot[0] == "load_offset" and _in_range(slot[1], node_range):
                tags.add("gather")
                break
        # Small-gather: flow vselect or addr-based bit extraction.
        if any(slot[0] == "vselect" for slot in instr.get("flow", [])):
            tags.add("small_gather")

        for engine, slots in instr.items():
            for slot in slots:
                op, dest, srcs = _slot_addrs(engine, slot)
                if dest is None:
                    continue
                # Small-gather bit extraction from addr-based indices.
                if op in ("-", "&", ">>") and _in_any(dest, [tmp1_range, tmp2_range, tmp4_range]):
                    if any(
                        _in_any(s, [addr_range, base1_range, base3_range]) for s in srcs
                    ):
                        tags.add("small_gather")
                # Addr update (writes addr pool or parity extraction).
                if _in_range(dest, addr_range):
                    tags.add("addr_update")
                if op == "&" and _in_range(dest, tmp1_range):
                    if any(_in_range(s, val_range) for s in srcs):
                        tags.add("addr_update")
                # Hash stages (writes val/tmp pools with hash ops).
                if op in hash_ops and (
                    _in_range(dest, val_range)
                    or _in_range(dest, tmp1_range)
                    or _in_range(dest, tmp2_range)
                ):
                    tags.add("hash")
                # Small-gather selection writes node pool using forest vectors.
                if _in_range(dest, node_range) and _in_any(dest, [node_range]):
                    if any(_in_any(s, forest_ranges) for s in srcs):
                        tags.add("small_gather")

        if not tags:
            tags.add("other")
        return tags

    def _bundle_depth_tags(instr):
        depth_tags = set()
        for engine, slots in instr.items():
            for slot in slots:
                op, dest, srcs = _slot_addrs(engine, slot)
                if op is None:
                    continue
                # Depth 0: XOR with vec_forest0.
                if op == "^" and dest is not None and _in_range(dest, val_range):
                    if any(_in_any(s, [forest0_range]) for s in srcs):
                        depth_tags.add("d0")
                if dest is not None and _in_any(dest, [forest0_range]):
                    depth_tags.add("d0")
                # Depth 1: vselect between forest1/2 or base1 addr math.
                if any(
                    _in_any(s, [forest1_range, forest2_range, base1_range]) for s in srcs
                ):
                    depth_tags.add("d1")
                if dest is not None and _in_any(dest, [forest1_range, forest2_range]):
                    depth_tags.add("d1")
                # Depth 2: vselect using forest3..6 or base3 addr math.
                if any(
                    _in_any(
                        s,
                        [
                            forest3_range,
                            forest4_range,
                            forest5_range,
                            forest6_range,
                            base3_range,
                        ],
                    )
                    for s in srcs
                ):
                    depth_tags.add("d2")
                if dest is not None and _in_any(
                    dest, [forest3_range, forest4_range, forest5_range, forest6_range]
                ):
                    depth_tags.add("d2")
                # Depth >=3: gather via load_offset into node pool.
                if (
                    engine == "load"
                    and op == "load_offset"
                    and dest is not None
                    and _in_range(dest, node_range)
                ):
                    depth_tags.add("d>=3")
                # Depth max: addr reset via vbroadcast from forest_values_p.
                if (
                    engine == "valu"
                    and op == "vbroadcast"
                    and dest is not None
                    and _in_range(dest, addr_range)
                ):
                    if srcs and _in_range(srcs[0], forest_values_p_range):
                        depth_tags.add("dmax")
        if not depth_tags:
            depth_tags.add("d?")
        return depth_tags

    under_by_tag = defaultdict(int)
    under_by_combo = defaultdict(int)
    under_total = 0
    under_by_depth = defaultdict(int)
    underused_bundles = []
    bundle_infos = []
    last_depth = None
    round_guess = 0
    seen_depth = False
    for instr in instrs[start:]:
        vslots = len(instr.get("valu", []))
        depth_tags = _bundle_depth_tags(instr)
        depth_primary = None
        if len(depth_tags) == 1:
            depth_primary = next(iter(depth_tags))
        if depth_primary is not None:
            if not seen_depth:
                round_guess = 0
                seen_depth = True
            if depth_primary == "d0" and last_depth not in (None, "d0"):
                round_guess += 1
            last_depth = depth_primary
        bundle_infos.append(
            {
                "instr": instr,
                "depth_tags": depth_tags,
                "depth_primary": depth_primary,
                "round": round_guess if seen_depth else None,
            }
        )
        if vslots not in (1, 2, 3):
            continue
        under_total += 1
        tags = _bundle_tags(instr)
        for tag in tags:
            under_by_tag[tag] += 1
        combo = tuple(sorted(tags))
        under_by_combo[combo] += 1
        underused_bundles.append((combo, instr))
        if depth_primary is None:
            under_by_depth["mixed"] += 1
        for dtag in depth_tags:
            under_by_depth[dtag] += 1

    print("Underused VALU bundles by phase tag (1-3 slots):")
    for tag, count in sorted(under_by_tag.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {tag}: {count}")
    print("Underused VALU bundle tag combinations (top 15):")
    for combo, count in sorted(under_by_combo.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
        print(f"  {combo}: {count}")

    print("Underused VALU bundles by depth tag (1-3 slots):")
    for tag, count in sorted(under_by_depth.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {tag}: {count}")

    # Show a few concrete bundles for the most common tag combinations.
    def _format_operand(val):
        if not isinstance(val, int):
            return val
        for addr, (name, length) in scratch_map.items():
            if addr <= val < addr + length:
                offset = val - addr
                if length == 1 or offset == 0:
                    return name
                return f"{name}+{offset}"
        return val

    def _format_slot(slot):
        op = slot[0]
        return tuple([op] + [_format_operand(v) for v in slot[1:]])

    max_examples = int(os.getenv("ANALYZE_PATTERNS_EXAMPLES", "5") or 5)
    print(f"Sample underused bundles by tag combo (up to {max_examples} each):")
    for combo, count in sorted(under_by_combo.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
        print(f"  {combo}: {count}")
        shown = 0
        for bcombo, instr in underused_bundles:
            if bcombo != combo:
                continue
            print("    bundle:")
            for engine in ("load", "alu", "valu", "flow", "store"):
                slots = instr.get(engine, [])
                if not slots:
                    continue
                formatted = [_format_slot(s) for s in slots]
                print(f"      {engine}: {formatted}")
            shown += 1
            if shown >= max_examples:
                break

    out_path = os.getenv("ANALYZE_PATTERNS_OUT", "pattern_bundles.txt").strip()
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("Bundle analysis (post-schedule)\n")
            f.write(f"start_bundle_index={start}\n")
            for idx, info in enumerate(bundle_infos):
                instr = info["instr"]
                vslots = len(instr.get("valu", []))
                aslots = len(instr.get("alu", []))
                lslots = len(instr.get("load", []))
                sslots = len(instr.get("store", []))
                fslots = len(instr.get("flow", []))
                under = 1 <= vslots <= 3
                depth_tags = ",".join(sorted(info["depth_tags"]))
                f.write(
                    f"[{idx}] round={info['round']} depth={info['depth_primary']} "
                    f"depth_tags={depth_tags} underused={under} "
                    f"occ=(v{vslots},a{aslots},l{lslots},s{sslots},f{fslots})\n"
                )
                for engine in ("load", "alu", "valu", "flow", "store"):
                    slots = instr.get(engine, [])
                    if not slots:
                        continue
                    formatted = [_format_slot(s) for s in slots]
                    f.write(f"  {engine}: {formatted}\n")
            f.write("end\n")
        print(f"Wrote full bundle dump to {out_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run kernel test.")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--prints", action="store_true")
    return parser.parse_args()


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
    do_kernel_test(
        args.forest_height,
        args.rounds,
        args.batch_size,
        seed=args.seed,
        trace=args.trace,
        prints=args.prints,
    )


def _has_cli_args(argv: list[str]) -> bool:
    cli_flags = {
        "--forest-height",
        "--rounds",
        "--batch-size",
        "--seed",
        "--trace",
        "--prints",
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
