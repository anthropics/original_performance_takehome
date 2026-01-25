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
"""
# Current best (1377 cycles):
START_OFFSETS=2,8,14,23,33,36,42,51,58,64,70,79,100,92,98,107,106,120,126,135,1,12,22,35,52,56,66,88,90,100,110,123 \
GROUPS_PER_BATCH=20 START_SPACING=7 START_SPACING_TAIL=11 PARITY_MASK=70 START_JITTER_PATTERN=2,1,0,2 python3 perf_takehome.py
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
    ops: list[Op],
    start_offsets: dict[int, int] | None = None,
    policy: str = "greedy",
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
    remaining_engine = None
    if policy == "balanced":
        remaining_engine = defaultdict(int)
        for op in ops:
            remaining_engine[op.engine] += 1
    users_count = [len(u) for u in users]
    priority_env = os.environ.get("SCHED_PRIORITY")
    if priority_env:
        order = [name.strip() for name in priority_env.split(",") if name.strip()]
        engine_priority = {name: idx for idx, name in enumerate(order)}
    else:
        engine_priority = {"valu": 0, "load": 1, "flow": 2, "alu": 3, "store": 4}
    op_rank = None
    if policy in ("critical", "ranked", "beam", "lookahead"):
        op_rank = [0] * len(ops)
        for op_id in range(len(ops) - 1, -1, -1):
            best = 0
            for user_id, latency in users[op_id]:
                cand = op_rank[user_id] + latency
                if cand > best:
                    best = cand
            op_rank[op_id] = best
    lookahead_depth = int(os.environ.get("LOOKAHEAD_DEPTH", "3"))
    beam_width = int(os.environ.get("BEAM_WIDTH", "3"))
    use_lookahead = policy in ("beam", "lookahead") and lookahead_depth > 0 and beam_width > 0
    score_policy = "greedy" if use_lookahead else policy
    total_slots_per_cycle = sum(engine_limits.values())

    def score_op(op_id: int, slots_used: dict[str, int], policy_override: str | None = None):
        pol = policy_override or policy
        op = ops[op_id]
        if slots_used[op.engine] >= engine_limits[op.engine]:
            return None
        if pol == "balanced" and remaining_engine is not None:
            work_score = -(
                remaining_engine[op.engine] / engine_limits.get(op.engine, 1)
            )
            return (work_score, -users_count[op_id])
        if op_rank is not None and pol in ("critical", "ranked"):
            if pol == "ranked":
                return (
                    engine_priority.get(op.engine, 99),
                    -op_rank[op_id],
                    -users_count[op_id],
                )
            return (
                -op_rank[op_id],
                engine_priority.get(op.engine, 99),
                -users_count[op_id],
            )
        return (engine_priority.get(op.engine, 99), -users_count[op_id])

    def rank_available(
        available: list[int], slots_used: dict[str, int], policy_override: str | None = None
    ) -> list[int]:
        scored: list[tuple[tuple, int]] = []
        for op_id in available:
            score = score_op(op_id, slots_used, policy_override=policy_override)
            if score is None:
                continue
            scored.append((score, op_id))
        scored.sort()
        return [op_id for _score, op_id in scored]

    def simulate_cycle(state: dict[str, object], first_choice: int | None = None):
        cycle_local = state["cycle"]
        ready_local = list(state["ready"])
        heapq.heapify(ready_local)
        available_local = list(state["available"])
        deps_local = list(state["deps_count"])
        earliest_local = list(state["earliest"])
        scheduled_count = state["scheduled_count"]
        slots_used_local = {k: 0 for k in engine_limits}
        scheduled_ops: list[int] = []

        def schedule_one(op_id: int) -> bool:
            nonlocal scheduled_count
            op = ops[op_id]
            if slots_used_local[op.engine] >= engine_limits[op.engine]:
                return False
            slots_used_local[op.engine] += 1
            scheduled_ops.append(op_id)
            scheduled_count += 1
            for user, latency in users[op_id]:
                deps_local[user] -= 1
                earliest_local[user] = max(earliest_local[user], cycle_local + latency)
                if deps_local[user] == 0:
                    if earliest_local[user] <= cycle_local:
                        available_local.append(user)
                    else:
                        heapq.heappush(ready_local, (earliest_local[user], user))
            return True

        if first_choice is not None:
            if first_choice in available_local:
                available_local.remove(first_choice)
            elif deps_local[first_choice] != 0 or earliest_local[first_choice] > cycle_local:
                return None
            if score_op(first_choice, slots_used_local, policy_override=score_policy) is None:
                return None
            if not schedule_one(first_choice):
                return None

        while available_local:
            ranked = rank_available(
                available_local, slots_used_local, policy_override=score_policy
            )
            if not ranked:
                break
            op_id = ranked[0]
            available_local.remove(op_id)
            if not schedule_one(op_id):
                break

        for op_id in available_local:
            heapq.heappush(ready_local, (earliest_local[op_id], op_id))

        next_cycle = cycle_local + 1
        if ready_local and ready_local[0][0] > next_cycle:
            next_cycle = ready_local[0][0]
        next_available = []
        while ready_local and ready_local[0][0] <= next_cycle:
            next_available.append(heapq.heappop(ready_local)[1])

        next_state = {
            "cycle": next_cycle,
            "ready": ready_local,
            "available": next_available,
            "deps_count": deps_local,
            "earliest": earliest_local,
            "scheduled_count": scheduled_count,
        }
        return scheduled_ops, next_state

    def select_lookahead_schedule(
        start_cycle: int,
        scheduled_so_far: int,
        ready_local: list[tuple[int, int]],
        available_local: list[int],
        deps_local: list[int],
        earliest_local: list[int],
    ) -> list[int] | None:
        state = {
            "cycle": start_cycle,
            "ready": list(ready_local),
            "available": list(available_local),
            "deps_count": list(deps_local),
            "earliest": list(earliest_local),
            "scheduled_count": 0,
        }
        beam = [state]

        def score_state(st: dict[str, object]):
            remaining = len(ops) - (scheduled_so_far + st["scheduled_count"])
            if remaining < 0:
                remaining = 0
            work_bound = (remaining + total_slots_per_cycle - 1) // total_slots_per_cycle
            cycle_offset = st["cycle"] - start_cycle
            return (cycle_offset + work_bound, -st["scheduled_count"])

        for depth in range(lookahead_depth):
            new_beam = []
            for st in beam:
                slots_used = {k: 0 for k in engine_limits}
                ranked = rank_available(
                    st["available"], slots_used, policy_override=score_policy
                )
                candidates = [None]
                for op_id in ranked[:beam_width]:
                    if op_id not in candidates:
                        candidates.append(op_id)
                for choice in candidates:
                    result = simulate_cycle(st, first_choice=choice)
                    if result is None:
                        continue
                    schedule, next_state = result
                    if depth == 0:
                        next_state["first_cycle_ops"] = schedule
                    else:
                        next_state["first_cycle_ops"] = st.get("first_cycle_ops", [])
                    new_beam.append(next_state)
            if not new_beam:
                break
            new_beam.sort(key=score_state)
            beam = new_beam[:beam_width]
        if not beam:
            return None
        best_state = min(beam, key=score_state)
        return best_state.get("first_cycle_ops")

    while scheduled < len(ops):
        if not ready:
            cycle += 1
            continue

        if ready[0][0] > cycle:
            cycle = ready[0][0]

        if cycle >= len(bundles):
            for _ in range(cycle - len(bundles) + 1):
                bundles.append(defaultdict(list))

        slots_used = {k: 0 for k in engine_limits}
        available: list[int] = []
        while ready and ready[0][0] <= cycle:
            available.append(heapq.heappop(ready)[1])

        def pick_op_index() -> int | None:
            best_idx = None
            best_score = None
            for i, op_id in enumerate(available):
                score = score_op(op_id, slots_used)
                if score is None:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = i
            return best_idx

        selected_ops = None
        if use_lookahead:
            selected_ops = select_lookahead_schedule(
                cycle, scheduled, ready, available, deps_count, earliest
            )
            if selected_ops == [] and available:
                selected_ops = None

        if selected_ops is not None:
            for op_id in selected_ops:
                op = ops[op_id]
                if op_id in available:
                    available.remove(op_id)
                bundles[cycle][op.engine].append(op.slot)
                slots_used[op.engine] += 1
                scheduled += 1
                if remaining_engine is not None:
                    remaining_engine[op.engine] -= 1
                for user, latency in users[op_id]:
                    deps_count[user] -= 1
                    earliest[user] = max(earliest[user], cycle + latency)
                    if deps_count[user] == 0:
                        if earliest[user] <= cycle:
                            available.append(user)
                        else:
                            heapq.heappush(ready, (earliest[user], user))
        else:
            while available:
                pick_idx = pick_op_index()
                if pick_idx is None:
                    break
                op_id = available.pop(pick_idx)
                op = ops[op_id]
                bundles[cycle][op.engine].append(op.slot)
                slots_used[op.engine] += 1
                scheduled += 1
                if remaining_engine is not None:
                    remaining_engine[op.engine] -= 1
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
        if not (
            8 <= forest_height <= 10
            and 8 <= rounds <= 20
            and 128 <= batch_size <= 256
            and batch_size % VLEN == 0
        ):
            return self.build_kernel_baseline(forest_height, n_nodes, batch_size, rounds)

        # Reset builder state
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        collect_ops = (
            getattr(self, "collect_ops", False) or os.environ.get("KERNEL_DEBUG_OPS") == "1"
        )
        if collect_ops:
            self.debug_ops = {"init": [], "batches": []}

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
        one_vec = vec_const(1)
        two_vec = vec_const(2)
        four_vec = vec_const(4)
        eight_vec = vec_const(8)
        one_scalar = scalar_const(1)
        batch_size_scalar = scalar_const(batch_size)
        forest_base_val = 7
        forest_base_scalar = scalar_const(forest_base_val)
        addr_base_scalar = scalar_const(forest_base_val + 15)
        neg6_scalar = scalar_const((2**32) - 6)
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
        node_addr = self.alloc_scratch("node_addr")
        emit_const(init_ops, node_addr, forest_base_val)
        for i in range(15):
            node_scalar = self.alloc_scratch(f"node_{i}")
            emit_load(init_ops, node_scalar, node_addr)
            node_vec_addr = self.alloc_scratch(f"node_{i}_v", VLEN)
            emit_vbroadcast(init_ops, node_vec_addr, node_scalar)
            node_vec[i] = node_vec_addr
            if i < 14:
                emit_alu(init_ops, "+", node_addr, node_addr, one_scalar)

        groups_per_batch = int(os.environ.get("GROUPS_PER_BATCH", "21"))
        start_spacing = int(os.environ.get("START_SPACING", "1"))
        start_period = int(os.environ.get("START_PERIOD", "0"))
        tail_spacing = int(os.environ.get("START_SPACING_TAIL", str(start_spacing)))
        jitter_pattern_text = os.environ.get("START_JITTER_PATTERN")
        jitter_pattern_tail_text = os.environ.get("START_JITTER_PATTERN_TAIL")
        start_offsets_text = os.environ.get("START_OFFSETS")
        phase_spacing = int(os.environ.get("PHASE_SPACING", "0"))
        phase_spacing_tail = int(os.environ.get("PHASE_SPACING_TAIL", str(phase_spacing)))
        phase_jitter_text = os.environ.get("PHASE_JITTER_PATTERN")
        phase_jitter_tail_text = os.environ.get("PHASE_JITTER_PATTERN_TAIL")
        jitter_pattern = None
        jitter_pattern_tail = None
        start_offsets_override = None
        phase_jitter_pattern = None
        phase_jitter_pattern_tail = None
        if jitter_pattern_text:
            jitter_pattern = [
                int(part.strip())
                for part in jitter_pattern_text.split(",")
                if part.strip()
            ]
        if jitter_pattern_tail_text:
            jitter_pattern_tail = [
                int(part.strip())
                for part in jitter_pattern_tail_text.split(",")
                if part.strip()
            ]
        if start_offsets_text:
            start_offsets_override = [
                int(part.strip())
                for part in start_offsets_text.split(",")
                if part.strip()
            ]
        if phase_jitter_text:
            phase_jitter_pattern = [
                int(part.strip())
                for part in phase_jitter_text.split(",")
                if part.strip()
            ]
        if phase_jitter_tail_text:
            phase_jitter_pattern_tail = [
                int(part.strip())
                for part in phase_jitter_tail_text.split(",")
                if part.strip()
            ]
        total_groups = batch_size // VLEN
        batches = (total_groups + groups_per_batch - 1) // groups_per_batch
        segment_len = forest_height + 1
        full_segments = rounds // segment_len
        remainder = rounds % segment_len

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
            if hash_shift_alu:
                sh19_scalar = scalar_const(19)
                for lane in range(VLEN):
                    emit_alu(op_list, ">>", tmp2 + lane, val + lane, sh19_scalar)
            else:
                emit_valu(op_list, ">>", tmp2, val, sh19_vec)
            emit_valu(op_list, "^", val, tmp1, tmp2)
            emit_muladd(op_list, val, val, m3_vec, c3_vec)
            emit_valu(op_list, "+", tmp1, val, c4_vec)
            if hash_shift_alu:
                sh9_scalar = scalar_const(9)
                for lane in range(VLEN):
                    emit_alu(op_list, "<<", tmp2 + lane, val + lane, sh9_scalar)
            else:
                emit_valu(op_list, "<<", tmp2, val, sh9_vec)
            emit_valu(op_list, "^", val, tmp1, tmp2)
            emit_muladd(op_list, val, val, m5_vec, c5_vec)
            emit_valu(op_list, "^", tmp1, val, c6_vec)
            if hash_shift_alu:
                sh16_scalar = scalar_const(16)
                for lane in range(VLEN):
                    emit_alu(op_list, ">>", tmp2 + lane, val + lane, sh16_scalar)
            else:
                emit_valu(op_list, ">>", tmp2, val, sh16_vec)
            emit_valu(op_list, "^", val, tmp1, tmp2)

        idx_add_alu = os.environ.get("IDX_ADD_ALU", "1") == "1"
        addr_muladd_alu = os.environ.get("ADDR_MULADD_ALU", "0") == "1"
        val_xor_alu = os.environ.get("VAL_XOR_ALU", "0") == "1"
        val_xor_top_alu = os.environ.get("VAL_XOR_TOP_ALU", "0") == "1"
        addr_update_flow = os.environ.get("ADDR_UPDATE_FLOW", "0") == "1"
        hash_shift_alu = os.environ.get("HASH_SHIFT_ALU", "0") == "1"
        idx_bias = os.environ.get("IDX_BIAS", "0") == "1"

        idx_base_scalar = addr_base_scalar
        load_bias_scalar = forest_base_scalar
        if idx_bias:
            idx_base_scalar = scalar_const(16)
            load_bias_scalar = scalar_const(forest_base_val - 1)

        def emit_val_xor(op_list: list[Op], dest: int, a1: int, a2: int):
            if val_xor_alu:
                for lane in range(VLEN):
                    emit_alu(op_list, "^", dest + lane, a1 + lane, a2 + lane)
            else:
                emit_valu(op_list, "^", dest, a1, a2)

        def emit_val_xor_top(op_list: list[Op], dest: int, a1: int, a2: int):
            if val_xor_top_alu:
                for lane in range(VLEN):
                    emit_alu(op_list, "^", dest + lane, a1 + lane, a2 + lane)
            else:
                emit_valu(op_list, "^", dest, a1, a2)

        def emit_addr_update(op_list: list[Op], addr: int, parity: int):
            if addr_muladd_alu:
                for lane in range(VLEN):
                    emit_alu(op_list, "+", addr + lane, addr + lane, addr + lane)
                    emit_alu(op_list, "+", addr + lane, addr + lane, parity + lane)
                    if not idx_bias:
                        emit_alu(op_list, "+", addr + lane, addr + lane, neg6_scalar)
            else:
                emit_muladd(op_list, addr, addr, two_vec, parity)
                if not idx_bias:
                    if idx_add_alu:
                        for lane in range(VLEN):
                            emit_alu(op_list, "+", addr + lane, addr + lane, neg6_scalar)
                    else:
                        emit_valu(op_list, "+", addr, addr, vec_const((2**32) - 6))

        def emit_addr_update_flow(op_list: list[Op], addr: int, parity: int, tmp: int):
            for lane in range(VLEN):
                emit_alu(op_list, "+", addr + lane, addr + lane, addr + lane)
                if not idx_bias:
                    emit_alu(op_list, "+", addr + lane, addr + lane, neg6_scalar)
                emit_alu(op_list, "+", tmp + lane, addr + lane, one_scalar)
            emit_vselect(op_list, addr, parity, tmp, addr)

        def emit_parity_scalar(op_list: list[Op], dest: int, src: int):
            for lane in range(VLEN):
                emit_alu(op_list, "&", dest + lane, src + lane, one_scalar)

        parity_mask = int(os.environ.get("PARITY_MASK", "246"), 0)
        parity_remaining_valu = os.environ.get("PARITY_REMAINING_VALU", "0") == "1"

        def emit_parity(op_list: list[Op], dest: int, src: int, mask_bit: int):
            if parity_mask & (1 << mask_bit):
                emit_parity_scalar(op_list, dest, src)
            else:
                emit_valu(op_list, "&", dest, src, one_vec)

        idx_bits_add_alu = os.environ.get("IDX_BITS_ADD_ALU", "1") == "1"
        idx_bits_alu = os.environ.get("IDX_BITS_ALU", "0") == "1"

        def emit_idx_from_bits(
            op_list: list[Op], idx: int, b0: int, b1: int, b2: int, tmp1: int, tmp2: int
        ):
            if idx_bits_alu:
                two_scalar = scalar_const(2)
                four_scalar = scalar_const(4)
                eight_scalar = scalar_const(8)
                for lane in range(VLEN):
                    emit_alu(op_list, "*", tmp1 + lane, b2 + lane, two_scalar)
                    emit_alu(op_list, "+", tmp1 + lane, tmp1 + lane, idx + lane)
                    emit_alu(op_list, "*", tmp2 + lane, b1 + lane, four_scalar)
                    emit_alu(op_list, "+", tmp1 + lane, tmp1 + lane, tmp2 + lane)
                    emit_alu(op_list, "*", tmp2 + lane, b0 + lane, eight_scalar)
                    emit_alu(op_list, "+", tmp1 + lane, tmp1 + lane, tmp2 + lane)
                    emit_alu(op_list, "+", idx + lane, tmp1 + lane, idx_base_scalar)
            else:
                emit_muladd(op_list, tmp1, b2, two_vec, idx)
                emit_muladd(op_list, tmp1, b1, four_vec, tmp1)
                emit_muladd(op_list, idx, b0, eight_vec, tmp1)
                if idx_bits_add_alu:
                    for lane in range(VLEN):
                        emit_alu(op_list, "+", idx + lane, idx + lane, idx_base_scalar)
                else:
                    emit_valu(op_list, "+", idx, idx, vec_const(int(idx_bias) + 15))

        load_addr_alu = os.environ.get("LOAD_ADDR_ALU", "1") == "1"
        sched_policy = os.environ.get("SCHED_POLICY", "greedy")

        def emit_load_node(op_list: list[Op], addr: int, node: int, addr_tmp: int):
            if idx_bias:
                for lane in range(VLEN):
                    emit_alu(op_list, "+", addr_tmp + lane, addr + lane, load_bias_scalar)
                for off in range(VLEN):
                    emit_load_offset(op_list, node, addr_tmp, off)
            else:
                for off in range(VLEN):
                    emit_load_offset(op_list, node, addr, off)

        for batch in range(batches):
            batch_ops: list[Op] = []
            group_start_ops: list[int] = []
            phase_start_ops: list[int | None] = []
            batch_groups = min(groups_per_batch, total_groups - batch * groups_per_batch)
            batch_spacing = start_spacing
            if batch_groups < groups_per_batch:
                batch_spacing = tail_spacing
            for g, vars_g in enumerate(group_vars[:batch_groups]):
                group_start_ops.append(len(batch_ops))
                phase_start_ops.append(None)
                group_id = batch * groups_per_batch + g
                base_addr_val = inp_base_val + group_id * VLEN
                emit_const(batch_ops, vars_g["base"], base_addr_val)
                emit_vload(batch_ops, vars_g["val"], vars_g["base"])
                if idx_bias:
                    emit_vbroadcast(batch_ops, vars_g["idx"], one_scalar)
                else:
                    emit_vbroadcast(batch_ops, vars_g["idx"], forest_base_scalar)

            for g, vars_g in enumerate(group_vars[:batch_groups]):
                val = vars_g["val"]
                idx = vars_g["idx"]
                tmp1 = vars_g["tmp1"]
                tmp2 = vars_g["tmp2"]
                b0 = vars_g["b0"]
                b1 = vars_g["b1"]
                b2 = vars_g["b2"]
                segments = []
                for seg in range(full_segments):
                    has_next = seg < full_segments - 1 or remainder > 0
                    segments.append((segment_len, has_next))
                if remainder:
                    segments.append((remainder, False))

                for seg_idx, (seg_rounds, has_next) in enumerate(segments):
                    parity_base = 0 if seg_idx % 2 == 0 else 4
                    top_rounds = min(4, seg_rounds)
                    remaining = seg_rounds - top_rounds
                    short_segment = seg_rounds < 4
                    full_segment = seg_rounds == segment_len

                    if top_rounds >= 1:
                        # Round 0 (root)
                        emit_val_xor_top(batch_ops, val, val, node_vec[0])
                        emit_hash(batch_ops, val, tmp1, tmp2)
                        emit_parity(batch_ops, b0, val, parity_base + 0)
                        if short_segment:
                            if addr_update_flow:
                                emit_addr_update_flow(batch_ops, idx, b0, tmp2)
                            else:
                                emit_addr_update(batch_ops, idx, b0)

                    if top_rounds >= 2:
                        # Round 1 (depth 1)
                        emit_vselect(batch_ops, tmp1, b0, node_vec[2], node_vec[1])
                        emit_val_xor_top(batch_ops, val, val, tmp1)
                        emit_hash(batch_ops, val, tmp1, tmp2)
                        emit_parity(batch_ops, b1, val, parity_base + 1)
                        if short_segment:
                            if addr_update_flow:
                                emit_addr_update_flow(batch_ops, idx, b1, tmp2)
                            else:
                                emit_addr_update(batch_ops, idx, b1)

                    if top_rounds >= 3:
                        # Round 2 (depth 2)
                        emit_vselect(batch_ops, tmp1, b1, node_vec[4], node_vec[3])
                        emit_vselect(batch_ops, tmp2, b1, node_vec[6], node_vec[5])
                        emit_vselect(batch_ops, tmp1, b0, tmp2, tmp1)
                        emit_val_xor_top(batch_ops, val, val, tmp1)
                        emit_hash(batch_ops, val, tmp1, tmp2)
                        emit_parity(batch_ops, b2, val, parity_base + 2)
                        if short_segment:
                            if addr_update_flow:
                                emit_addr_update_flow(batch_ops, idx, b2, tmp2)
                            else:
                                emit_addr_update(batch_ops, idx, b2)

                    if top_rounds >= 4:
                        # Round 3 (depth 3)
                        emit_vselect(batch_ops, tmp1, b2, node_vec[8], node_vec[7])
                        emit_vselect(batch_ops, tmp2, b2, node_vec[10], node_vec[9])
                        emit_vselect(batch_ops, tmp1, b1, tmp2, tmp1)
                        emit_vselect(batch_ops, tmp2, b2, node_vec[12], node_vec[11])
                        emit_vselect(batch_ops, idx, b2, node_vec[14], node_vec[13])
                        emit_vselect(batch_ops, tmp2, b1, idx, tmp2)
                        emit_vselect(batch_ops, tmp1, b0, tmp2, tmp1)
                        emit_val_xor_top(batch_ops, val, val, tmp1)
                        emit_hash(batch_ops, val, tmp1, tmp2)
                        emit_parity(batch_ops, idx, val, parity_base + 3)
                        emit_idx_from_bits(batch_ops, idx, b0, b1, b2, tmp1, tmp2)

                    if remaining > 0:
                        for r in range(remaining):
                            if r == 0:
                                phase_start_ops[g] = len(batch_ops)
                            emit_load_node(batch_ops, idx, tmp1, tmp2)
                            emit_val_xor(batch_ops, val, val, tmp1)
                            emit_hash(batch_ops, val, tmp1, tmp2)
                            if r == remaining - 1:
                                if full_segment:
                                    if idx_bias:
                                        emit_vbroadcast(batch_ops, idx, one_scalar)
                                    else:
                                        emit_vbroadcast(batch_ops, idx, forest_base_scalar)
                                else:
                                    if parity_remaining_valu:
                                        emit_valu(batch_ops, "&", tmp1, val, one_vec)
                                    else:
                                        emit_parity_scalar(batch_ops, tmp1, val)
                                    if addr_update_flow:
                                        emit_addr_update_flow(batch_ops, idx, tmp1, tmp2)
                                    else:
                                        emit_addr_update(batch_ops, idx, tmp1)
                            else:
                                if parity_remaining_valu:
                                    emit_valu(batch_ops, "&", tmp1, val, one_vec)
                                else:
                                    emit_parity_scalar(batch_ops, tmp1, val)
                                if addr_update_flow:
                                    emit_addr_update_flow(batch_ops, idx, tmp1, tmp2)
                                else:
                                    emit_addr_update(batch_ops, idx, tmp1)

                emit_vstore(batch_ops, vars_g["base"], val)
                for lane in range(VLEN):
                    if idx_bias:
                        emit_alu(batch_ops, "-", idx + lane, idx + lane, one_scalar)
                    else:
                        emit_alu(batch_ops, "-", idx + lane, idx + lane, forest_base_scalar)
                emit_alu(batch_ops, "-", vars_g["base"], vars_g["base"], batch_size_scalar)
                emit_vstore(batch_ops, vars_g["base"], idx)

            pattern = jitter_pattern
            if batch_groups < groups_per_batch and jitter_pattern_tail is not None:
                pattern = jitter_pattern_tail
            if start_period > 0:
                start_offsets = {
                    op_id: (gi % start_period) * batch_spacing
                    for gi, op_id in enumerate(group_start_ops)
                }
            else:
                start_offsets = {
                    op_id: gi * batch_spacing for gi, op_id in enumerate(group_start_ops)
                }
            if pattern:
                for gi, op_id in enumerate(group_start_ops):
                    start_offsets[op_id] += pattern[gi % len(pattern)]
            if phase_spacing > 0:
                phase_pattern = phase_jitter_pattern
                if batch_groups < groups_per_batch and phase_jitter_pattern_tail is not None:
                    phase_pattern = phase_jitter_pattern_tail
                phase_stride = phase_spacing
                if batch_groups < groups_per_batch:
                    phase_stride = phase_spacing_tail
                for gi, op_id in enumerate(phase_start_ops):
                    if op_id is None:
                        continue
                    offset = gi * phase_stride
                    if phase_pattern:
                        offset += phase_pattern[gi % len(phase_pattern)]
                    start_offsets[op_id] = max(start_offsets.get(op_id, 0), offset)
            if start_offsets_override:
                base_index = batch * groups_per_batch
                for gi, op_id in enumerate(group_start_ops):
                    override_idx = base_index + gi
                    if override_idx < len(start_offsets_override):
                        start_offsets[op_id] = start_offsets_override[override_idx]
            if collect_ops:
                self.debug_ops["batches"].append(
                    {
                        "ops": batch_ops,
                        "group_start_ops": list(group_start_ops),
                        "start_offsets": dict(start_offsets),
                    }
                )
            self.instrs.extend(
                schedule_ops(batch_ops, start_offsets=start_offsets, policy=sched_policy)
            )

        if collect_ops:
            self.debug_ops["init"] = init_ops
        init_instrs = schedule_ops(init_ops, policy=sched_policy)
        self.instrs = [{"flow": [("pause",)]}] + init_instrs + self.instrs
        self.instrs.append({"flow": [("pause",)]})

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
