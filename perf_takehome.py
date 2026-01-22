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

from collections import defaultdict, deque
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


class KernelBuilder:
    def __init__(self):
        self.slots = []
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _addr_range(self, base, length):
        return set(range(base, base + length))

    def _is_barrier(self, engine, slot):
        if engine != "flow":
            return False
        return slot[0] in (
            "pause",
            "halt",
            "jump",
            "jump_indirect",
            "cond_jump",
            "cond_jump_rel",
        )

    def _slot_rw(self, engine, slot):
        reads = set()
        writes = set()
        mem_load = False
        mem_store = False
        if engine == "alu":
            _, dest, a1, a2 = slot
            reads.update((a1, a2))
            writes.add(dest)
        elif engine == "valu":
            op = slot[0]
            if op == "vbroadcast":
                _, dest, src = slot
                reads.add(src)
                writes.update(self._addr_range(dest, VLEN))
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                reads.update(self._addr_range(a, VLEN))
                reads.update(self._addr_range(b, VLEN))
                reads.update(self._addr_range(c, VLEN))
                writes.update(self._addr_range(dest, VLEN))
            else:
                _, dest, a1, a2 = slot
                reads.update(self._addr_range(a1, VLEN))
                reads.update(self._addr_range(a2, VLEN))
                writes.update(self._addr_range(dest, VLEN))
        elif engine == "load":
            op = slot[0]
            if op == "const":
                _, dest, _ = slot
                writes.add(dest)
            elif op == "load":
                _, dest, addr = slot
                reads.add(addr)
                writes.add(dest)
                mem_load = True
            elif op == "load_offset":
                _, dest, addr, offset = slot
                reads.add(addr + offset)
                writes.add(dest + offset)
                mem_load = True
            elif op == "vload":
                _, dest, addr = slot
                reads.add(addr)
                writes.update(self._addr_range(dest, VLEN))
                mem_load = True
        elif engine == "store":
            op = slot[0]
            if op == "store":
                _, addr, src = slot
                reads.update((addr, src))
                mem_store = True
            elif op == "vstore":
                _, addr, src = slot
                reads.add(addr)
                reads.update(self._addr_range(src, VLEN))
                mem_store = True
        elif engine == "flow":
            op = slot[0]
            if op == "select":
                _, dest, cond, a, b = slot
                reads.update((cond, a, b))
                writes.add(dest)
            elif op == "add_imm":
                _, dest, a, _ = slot
                reads.add(a)
                writes.add(dest)
            elif op == "vselect":
                _, dest, cond, a, b = slot
                reads.update(self._addr_range(cond, VLEN))
                reads.update(self._addr_range(a, VLEN))
                reads.update(self._addr_range(b, VLEN))
                writes.update(self._addr_range(dest, VLEN))
            elif op == "trace_write":
                _, val = slot
                reads.add(val)
            elif op == "coreid":
                _, dest = slot
                writes.add(dest)
            elif op == "cond_jump":
                _, cond, _ = slot
                reads.add(cond)
            elif op == "cond_jump_rel":
                _, cond, _ = slot
                reads.add(cond)
            elif op == "jump_indirect":
                _, addr = slot
                reads.add(addr)
        return reads, writes, mem_load, mem_store

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        cur = defaultdict(list)
        cur_reads = set()
        cur_writes = set()
        cur_counts = defaultdict(int)
        cur_load = False
        cur_store = False

        def flush():
            nonlocal cur, cur_reads, cur_writes, cur_counts, cur_load, cur_store
            if cur:
                instrs.append(dict(cur))
            cur = defaultdict(list)
            cur_reads = set()
            cur_writes = set()
            cur_counts = defaultdict(int)
            cur_load = False
            cur_store = False

        for engine, slot in slots:
            if engine == "debug" or self._is_barrier(engine, slot):
                flush()
                instrs.append({engine: [slot]})
                continue

            reads, writes, mem_load, mem_store = self._slot_rw(engine, slot)

            if cur_counts[engine] >= SLOT_LIMITS[engine]:
                flush()
            if (mem_load and cur_store) or (mem_store and cur_load):
                flush()
            if reads & cur_writes or writes & (cur_reads | cur_writes):
                flush()

            cur[engine].append(slot)
            cur_counts[engine] += 1
            cur_reads.update(reads)
            cur_writes.update(writes)
            if mem_load:
                cur_load = True
            if mem_store:
                cur_store = True

        flush()
        return instrs

    def add(self, engine, slot):
        self.slots.append((engine, slot))

    def flush_slots(self):
        if self.slots:
            self.instrs.extend(self.build(self.slots))
            self.slots = []

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

    def build_hash(self, val_hash_addr, tmp1, tmp2):
        slots = []

        # Use multiply/add for the shift-add stages to reduce ops.
        mul_map = {
            12: 4097,  # (1 + 2^12)
            5: 33,  # (1 + 2^5)
            3: 9,  # (1 + 2^3)
        }

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<" and val3 in mul_map:
                mul_const = self.scratch_const(mul_map[val3])
                slots.append(("alu", ("*", tmp1, val_hash_addr, mul_const)))
                slots.append(("alu", ("+", val_hash_addr, tmp1, self.scratch_const(val1))))
            else:
                slots.append(
                    ("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1)))
                )
                slots.append(
                    ("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3)))
                )
                slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))

        return slots

    def build_hash_v(self, val_hash_addr, tmp1, tmp2, vconsts):
        slots = []
        # Stages 1, 3, 5 are shift-adds that can be reduced to multiply_add.
        slots.append(
            (
                "valu",
                (
                    "multiply_add",
                    val_hash_addr,
                    val_hash_addr,
                    vconsts["mul_4097"],
                    vconsts["c1_0"],
                ),
            )
        )
        slots.append(("valu", ("^", tmp1, val_hash_addr, vconsts["c1_1"])))
        slots.append(("valu", (">>", tmp2, val_hash_addr, vconsts["shr_19"])))
        slots.append(("valu", ("^", val_hash_addr, tmp1, tmp2)))
        slots.append(
            (
                "valu",
                (
                    "multiply_add",
                    val_hash_addr,
                    val_hash_addr,
                    vconsts["mul_33"],
                    vconsts["c1_2"],
                ),
            )
        )
        slots.append(("valu", ("+", tmp1, val_hash_addr, vconsts["c1_3"])))
        slots.append(("valu", ("<<", tmp2, val_hash_addr, vconsts["shl_9"])))
        slots.append(("valu", ("^", val_hash_addr, tmp1, tmp2)))
        slots.append(
            (
                "valu",
                (
                    "multiply_add",
                    val_hash_addr,
                    val_hash_addr,
                    vconsts["mul_9"],
                    vconsts["c1_4"],
                ),
            )
        )
        slots.append(("valu", ("^", tmp1, val_hash_addr, vconsts["c1_5"])))
        slots.append(("valu", (">>", tmp2, val_hash_addr, vconsts["shr_16"])))
        slots.append(("valu", ("^", val_hash_addr, tmp1, tmp2)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized implementation with simple VLIW slot packing.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr_val = self.alloc_scratch("tmp_addr_val")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
            "header_pad",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # Load header [0..7] in one vload (rounds..header_pad are contiguous).
        self.add("load", ("const", tmp1, 0))
        self.add("load", ("vload", self.scratch["rounds"], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        per_core_batch = batch_size

        # Scalar scratch registers (tail path)
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)
        v_root = self.alloc_scratch("v_root", VLEN)
        v_node2 = self.alloc_scratch("v_node2", VLEN)
        v_diff12 = self.alloc_scratch("v_diff12", VLEN)
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        root_node = self.alloc_scratch("root_node")
        node1 = self.alloc_scratch("node1")
        node2 = self.alloc_scratch("node2")
        node3 = self.alloc_scratch("node3")
        node4 = self.alloc_scratch("node4")
        node5 = self.alloc_scratch("node5")
        node6 = self.alloc_scratch("node6")
        node7 = self.alloc_scratch("node7")
        self.add("alu", ("+", tmp_addr, self.scratch["forest_values_p"], zero_const))
        self.add("load", ("vload", root_node, tmp_addr))
        self.add("valu", ("vbroadcast", v_root, root_node))
        self.add("valu", ("vbroadcast", v_node2, node2))
        self.add("valu", ("vbroadcast", v_diff12, node1))
        self.add("valu", ("-", v_diff12, v_diff12, v_node2))
        v_node3 = self.alloc_scratch("v_node3", VLEN)
        v_node4 = self.alloc_scratch("v_node4", VLEN)
        v_node5 = self.alloc_scratch("v_node5", VLEN)
        v_node6 = self.alloc_scratch("v_node6", VLEN)
        self.add("valu", ("vbroadcast", v_node3, node3))
        self.add("valu", ("vbroadcast", v_node4, node4))
        self.add("valu", ("vbroadcast", v_node5, node5))
        self.add("valu", ("vbroadcast", v_node6, node6))

        vconsts = {
            "mul_4097": self.alloc_scratch("v_mul_4097", VLEN),
            "mul_33": self.alloc_scratch("v_mul_33", VLEN),
            "mul_9": self.alloc_scratch("v_mul_9", VLEN),
            "c1_0": self.alloc_scratch("v_c1_0", VLEN),
            "c1_1": self.alloc_scratch("v_c1_1", VLEN),
            "c1_2": self.alloc_scratch("v_c1_2", VLEN),
            "c1_3": self.alloc_scratch("v_c1_3", VLEN),
            "c1_4": self.alloc_scratch("v_c1_4", VLEN),
            "c1_5": self.alloc_scratch("v_c1_5", VLEN),
            "shr_19": self.alloc_scratch("v_shr_19", VLEN),
            "shl_9": self.alloc_scratch("v_shl_9", VLEN),
            "shr_16": self.alloc_scratch("v_shr_16", VLEN),
        }

        self.add("valu", ("vbroadcast", vconsts["mul_4097"], self.scratch_const(4097)))
        self.add("valu", ("vbroadcast", vconsts["mul_33"], self.scratch_const(33)))
        self.add("valu", ("vbroadcast", vconsts["mul_9"], self.scratch_const(9)))
        self.add(
            "valu",
            ("vbroadcast", vconsts["c1_0"], self.scratch_const(0x7ED55D16)),
        )
        self.add(
            "valu",
            ("vbroadcast", vconsts["c1_1"], self.scratch_const(0xC761C23C)),
        )
        self.add(
            "valu",
            ("vbroadcast", vconsts["c1_2"], self.scratch_const(0x165667B1)),
        )
        self.add(
            "valu",
            ("vbroadcast", vconsts["c1_3"], self.scratch_const(0xD3A2646C)),
        )
        self.add(
            "valu",
            ("vbroadcast", vconsts["c1_4"], self.scratch_const(0xFD7046C5)),
        )
        self.add(
            "valu",
            ("vbroadcast", vconsts["c1_5"], self.scratch_const(0xB55A4F09)),
        )
        self.add("valu", ("vbroadcast", vconsts["shr_19"], self.scratch_const(19)))
        self.add("valu", ("vbroadcast", vconsts["shl_9"], self.scratch_const(9)))
        self.add("valu", ("vbroadcast", vconsts["shr_16"], self.scratch_const(16)))

        # Pause instructions are matched up with yield statements in the reference kernel.
        self.add("flow", ("pause",))

        total_vec_count = per_core_batch // VLEN
        vec_count = total_vec_count
        vec_batch = total_vec_count * VLEN

        v_idx = [self.alloc_scratch(f"v_idx_{u}", VLEN) for u in range(vec_count)]
        v_val = [self.alloc_scratch(f"v_val_{u}", VLEN) for u in range(vec_count)]
        v_node = [self.alloc_scratch(f"v_node_{u}", VLEN) for u in range(vec_count)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{u}", VLEN) for u in range(vec_count)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{u}", VLEN) for u in range(vec_count)]
        addr_vals = [self.alloc_scratch(f"addr_val_{u}") for u in range(vec_count)]

        rounds_period = forest_height + 1
        vlen_const = self.scratch_const(VLEN)
        def compute_addr_vals(block_start, block_n):
            if block_start == 0:
                self.add(
                    "alu",
                    ("+", addr_vals[0], self.scratch["inp_values_p"], zero_const),
                )
            else:
                self.add(
                    "alu",
                    (
                        "+",
                        addr_vals[0],
                        self.scratch["inp_values_p"],
                        self.scratch_const(block_start * VLEN),
                    ),
                )
            for u in range(1, block_n):
                self.add("alu", ("+", addr_vals[u], addr_vals[u - 1], vlen_const))
            self.flush_slots()

        def emit_vec_alu(tasks, op, dest, a1, a2):
            for lane in range(VLEN):
                tasks.append(("alu", (op, dest + lane, a1 + lane, a2 + lane)))

        def emit_hash_ops(tasks, u):
            tasks.append(
                ("valu", ("multiply_add", v_val[u], v_val[u], vconsts["mul_4097"], vconsts["c1_0"]))
            )
            tasks.append(("valu", (">>", v_node[u], v_val[u], vconsts["shr_19"])))
            tasks.append(("valu", ("^", v_val[u], v_val[u], v_node[u])))
            emit_vec_alu(tasks, "^", v_val[u], v_val[u], vconsts["c1_1"])
            tasks.append(
                ("valu", ("multiply_add", v_val[u], v_val[u], vconsts["mul_33"], vconsts["c1_2"]))
            )
            tasks.append(("valu", ("<<", v_node[u], v_val[u], vconsts["shl_9"])))
            if os.getenv("HASH_ADD_ALU", "0") == "1":
                emit_vec_alu(tasks, "+", v_val[u], v_val[u], vconsts["c1_3"])
            else:
                tasks.append(("valu", ("+", v_val[u], v_val[u], vconsts["c1_3"])))
            tasks.append(("valu", ("^", v_val[u], v_val[u], v_node[u])))
            tasks.append(
                ("valu", ("multiply_add", v_val[u], v_val[u], vconsts["mul_9"], vconsts["c1_4"]))
            )
            tasks.append(("valu", (">>", v_node[u], v_val[u], vconsts["shr_16"])))
            emit_vec_alu(tasks, "^", v_val[u], v_val[u], v_node[u])
            emit_vec_alu(tasks, "^", v_val[u], v_val[u], vconsts["c1_5"])

        def build_stream_tasks(block_n):
            stream_tasks = []
            for u in range(block_n):
                tasks = []
                tasks.append(("load", ("vload", v_val[u], addr_vals[u])))
                for r in range(rounds):
                    depth = r % rounds_period
                    if depth == 0:
                        tasks.append(("valu", ("^", v_val[u], v_val[u], v_root)))
                        emit_hash_ops(tasks, u)
                        tasks.append(("valu", ("&", v_node[u], v_val[u], v_one)))
                        tasks.append(("valu", ("+", v_idx[u], v_node[u], v_one)))
                    elif depth == 1:
                        tasks.append(("valu", ("&", v_node[u], v_idx[u], v_one)))
                        tasks.append(
                            ("valu", ("multiply_add", v_node[u], v_node[u], v_diff12, v_node2))
                        )
                        tasks.append(("valu", ("^", v_val[u], v_val[u], v_node[u])))
                        emit_hash_ops(tasks, u)
                        tasks.append(("valu", ("&", v_node[u], v_val[u], v_one)))
                        tasks.append(("valu", ("+", v_node[u], v_node[u], v_one)))
                        tasks.append(
                            ("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_node[u]))
                        )
                    elif depth == 2:
                        # Use cached nodes 3..6, select via vselect to avoid 8 loads.
                        # Use idx bits directly: (b1 b0) = idx & (2,1).
                        tasks.append(("valu", ("&", v_tmp1[u], v_idx[u], v_one)))  # b0
                        tasks.append(("valu", ("&", v_tmp2[u], v_idx[u], v_two)))  # b1 (0/2)
                        tasks.append(("flow", ("vselect", v_node[u], v_tmp1[u], v_node5, v_node4)))
                        tasks.append(("flow", ("vselect", v_tmp1[u], v_tmp1[u], v_node3, v_node6)))
                        tasks.append(("flow", ("vselect", v_node[u], v_tmp2[u], v_tmp1[u], v_node[u])))
                        tasks.append(("valu", ("^", v_val[u], v_val[u], v_node[u])))
                        emit_hash_ops(tasks, u)
                        tasks.append(("valu", ("&", v_node[u], v_val[u], v_one)))
                        tasks.append(("valu", ("+", v_node[u], v_node[u], v_one)))
                        tasks.append(
                            ("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_node[u]))
                        )
                    else:
                        tasks.append(("valu", ("+", v_node[u], v_idx[u], v_forest_base)))
                        for lane in range(VLEN):
                            tasks.append(("load", ("load_offset", v_node[u], v_node[u], lane)))
                        tasks.append(("valu", ("^", v_val[u], v_val[u], v_node[u])))
                        emit_hash_ops(tasks, u)
                        if depth == forest_height:
                            tasks.append(("valu", ("+", v_idx[u], v_zero, v_zero)))
                        else:
                            tasks.append(("valu", ("&", v_node[u], v_val[u], v_one)))
                            tasks.append(("valu", ("+", v_node[u], v_node[u], v_one)))
                            tasks.append(
                                ("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_node[u]))
                            )
                stream_tasks.append(deque(tasks))
            return stream_tasks

        def schedule_streams(stream_tasks):
            n_streams = len(stream_tasks)
            valu_queue = deque()
            load_queue = deque()
            flow_queue = deque()
            alu_queue = deque()

            # Stagger stream start times to desynchronize load-free rounds.
            stagger_groups = int(os.getenv("STAGGER_GROUPS", "9"))
            stagger_step = int(os.getenv("STAGGER_STEP", "35"))
            stagger_mode = os.getenv("STAGGER_MODE", "groups")
            stagger_list = os.getenv("STAGGER_LIST")
            if stagger_list:
                offsets = [int(x) for x in stagger_list.split(",") if x]
                if offsets:
                    if len(offsets) < n_streams:
                        offsets.extend([offsets[-1]] * (n_streams - len(offsets)))
                    start_offsets = offsets[:n_streams]
                else:
                    start_offsets = [0] * n_streams
            elif stagger_mode == "linear":
                start_offsets = [u * stagger_step for u in range(n_streams)]
            elif stagger_mode == "mod":
                stagger_period = int(os.getenv("STAGGER_PERIOD", "200"))
                start_offsets = [
                    (u * stagger_step) % stagger_period for u in range(n_streams)
                ]
            else:
                start_offsets = [
                    (u % stagger_groups) * stagger_step for u in range(n_streams)
                ]
            pending_streams = set(range(n_streams))

            def enqueue_stream(u):
                if stream_tasks[u]:
                    head = stream_tasks[u][0][0]
                    if head == "valu":
                        valu_queue.append(u)
                    elif head == "load":
                        load_queue.append(u)
                    elif head == "flow":
                        flow_queue.append(u)
                    elif head == "alu":
                        alu_queue.append(u)
                    else:
                        raise RuntimeError(f"Unknown head op {head}")

            def activate_ready(cur_cycle):
                ready = [u for u in pending_streams if start_offsets[u] <= cur_cycle]
                for u in ready:
                    pending_streams.remove(u)
                    enqueue_stream(u)

            active_valu = []
            active_alu = []

            def refill_active():
                while len(active_valu) < SLOT_LIMITS["valu"] and valu_queue:
                    u = valu_queue.popleft()
                    if u not in active_valu:
                        active_valu.append(u)

            def refill_active_alu():
                while len(active_alu) < SLOT_LIMITS["alu"] and alu_queue:
                    u = alu_queue.popleft()
                    if u not in active_alu:
                        active_alu.append(u)

            refill_active()
            refill_active_alu()

            while any(stream_tasks):
                activate_ready(len(self.instrs))
                if (
                    not load_queue
                    and not valu_queue
                    and not flow_queue
                    and not alu_queue
                    and pending_streams
                ):
                    # Ensure progress if all queues are empty.
                    u = min(pending_streams, key=lambda x: start_offsets[x])
                    pending_streams.remove(u)
                    enqueue_stream(u)

                bundle = {}
                cur_reads = set()
                cur_writes = set()
                cur_counts = defaultdict(int)
                cur_load = False
                cur_store = False

                def try_add(engine, slot):
                    nonlocal cur_reads, cur_writes, cur_counts, cur_load, cur_store
                    reads, writes, mem_load, mem_store = self._slot_rw(engine, slot)
                    allow_war = os.getenv("ALLOW_WAR", "0") == "1"
                    if cur_counts[engine] >= SLOT_LIMITS[engine]:
                        return False
                    if (mem_load and cur_store) or (mem_store and cur_load):
                        return False
                    if reads & cur_writes:
                        return False
                    if allow_war:
                        # Allow write-after-read in the same cycle; writes still
                        # conflict with earlier writes.
                        if writes & cur_writes:
                            return False
                    else:
                        if writes & (cur_reads | cur_writes):
                            return False
                    bundle.setdefault(engine, []).append(slot)
                    cur_counts[engine] += 1
                    cur_reads.update(reads)
                    cur_writes.update(writes)
                    if mem_load:
                        cur_load = True
                    if mem_store:
                        cur_store = True
                    return True

                def schedule_loads():
                    load_slots = 0
                    load_spread = os.getenv("LOAD_SPREAD", "0") == "1"

                    if load_spread:
                        # Schedule at most one load per stream to spread load phases.
                        pending = len(load_queue)
                        while load_slots < SLOT_LIMITS["load"] and load_queue and pending > 0:
                            u = load_queue.popleft()
                            pending -= 1
                            if stream_tasks[u] and stream_tasks[u][0][0] == "load":
                                if try_add("load", stream_tasks[u][0][1]):
                                    stream_tasks[u].popleft()
                                    load_slots += 1
                            if stream_tasks[u]:
                                next_head = stream_tasks[u][0][0]
                                if next_head == "load":
                                    load_queue.append(u)
                                elif next_head == "valu":
                                    valu_queue.append(u)
                                elif next_head == "flow":
                                    flow_queue.append(u)
                                elif next_head == "alu":
                                    alu_queue.append(u)
                                else:
                                    raise RuntimeError(f"Unknown head op {next_head}")
                    else:
                        # Schedule up to 2 loads, preferring to drain a burst from one stream,
                        # but filling a second slot from another stream if available.
                        if load_queue:
                            u = load_queue.popleft()
                            while (
                                stream_tasks[u]
                                and stream_tasks[u][0][0] == "load"
                                and load_slots < SLOT_LIMITS["load"]
                            ):
                                if try_add("load", stream_tasks[u][0][1]):
                                    stream_tasks[u].popleft()
                                    load_slots += 1
                                else:
                                    break
                            if stream_tasks[u]:
                                next_head = stream_tasks[u][0][0]
                                if next_head == "load":
                                    load_queue.appendleft(u)
                                elif next_head == "valu":
                                    valu_queue.append(u)
                                elif next_head == "flow":
                                    flow_queue.append(u)
                                elif next_head == "alu":
                                    alu_queue.append(u)
                                else:
                                    raise RuntimeError(f"Unknown head op {next_head}")

                        if load_slots < SLOT_LIMITS["load"] and load_queue:
                            pending = len(load_queue)
                            while (
                                load_slots < SLOT_LIMITS["load"]
                                and load_queue
                                and pending > 0
                            ):
                                v = load_queue.popleft()
                                pending -= 1
                                if stream_tasks[v] and stream_tasks[v][0][0] == "load":
                                    if try_add("load", stream_tasks[v][0][1]):
                                        stream_tasks[v].popleft()
                                        load_slots += 1
                                if stream_tasks[v]:
                                    next_head = stream_tasks[v][0][0]
                                    if next_head == "load":
                                        load_queue.append(v)
                                    elif next_head == "valu":
                                        valu_queue.append(v)
                                    elif next_head == "flow":
                                        flow_queue.append(v)
                                    elif next_head == "alu":
                                        alu_queue.append(v)
                                    else:
                                        raise RuntimeError(f"Unknown head op {next_head}")

                schedule_loads()

                # Schedule up to 1 flow op.
                if flow_queue:
                    pending_flow = len(flow_queue)
                    while flow_queue and pending_flow > 0:
                        u = flow_queue.popleft()
                        pending_flow -= 1
                        if stream_tasks[u] and stream_tasks[u][0][0] == "flow":
                            if try_add("flow", stream_tasks[u][0][1]):
                                stream_tasks[u].popleft()
                        if stream_tasks[u]:
                            next_head = stream_tasks[u][0][0]
                            if next_head == "load":
                                load_queue.append(u)
                            elif next_head == "valu":
                                valu_queue.append(u)
                            elif next_head == "flow":
                                flow_queue.append(u)
                            elif next_head == "alu":
                                alu_queue.append(u)
                            else:
                                raise RuntimeError(f"Unknown head op {next_head}")
                        break

                # Schedule up to 12 alu ops from active streams.
                new_active_alu = []
                for u in active_alu:
                    if len(bundle.get("alu", [])) >= SLOT_LIMITS["alu"]:
                        new_active_alu.append(u)
                        continue
                    while (
                        stream_tasks[u]
                        and stream_tasks[u][0][0] == "alu"
                        and len(bundle.get("alu", [])) < SLOT_LIMITS["alu"]
                    ):
                        if try_add("alu", stream_tasks[u][0][1]):
                            stream_tasks[u].popleft()
                            continue
                        break
                    if stream_tasks[u] and stream_tasks[u][0][0] == "alu":
                        new_active_alu.append(u)
                    else:
                        if stream_tasks[u]:
                            next_head = stream_tasks[u][0][0]
                            if next_head == "load":
                                load_queue.append(u)
                            elif next_head == "valu":
                                valu_queue.append(u)
                            elif next_head == "flow":
                                flow_queue.append(u)
                            elif next_head == "alu":
                                alu_queue.append(u)
                            else:
                                raise RuntimeError(f"Unknown head op {next_head}")

                # If we still have alu capacity, pull more streams this cycle.
                pending = len(alu_queue)
                while (
                    len(bundle.get("alu", [])) < SLOT_LIMITS["alu"]
                    and alu_queue
                    and pending > 0
                ):
                    u = alu_queue.popleft()
                    pending -= 1
                    while (
                        stream_tasks[u]
                        and stream_tasks[u][0][0] == "alu"
                        and len(bundle.get("alu", [])) < SLOT_LIMITS["alu"]
                    ):
                        if try_add("alu", stream_tasks[u][0][1]):
                            stream_tasks[u].popleft()
                            continue
                        break
                    if stream_tasks[u] and stream_tasks[u][0][0] == "alu":
                        new_active_alu.append(u)
                    else:
                        if stream_tasks[u]:
                            next_head = stream_tasks[u][0][0]
                            if next_head == "load":
                                load_queue.append(u)
                            elif next_head == "valu":
                                valu_queue.append(u)
                            elif next_head == "flow":
                                flow_queue.append(u)
                            elif next_head == "alu":
                                alu_queue.append(u)
                            else:
                                raise RuntimeError(f"Unknown head op {next_head}")

                # Schedule up to 6 valu ops from active streams.
                new_active = []
                for u in active_valu:
                    if len(bundle.get("valu", [])) >= SLOT_LIMITS["valu"]:
                        new_active.append(u)
                        continue
                    if (
                        stream_tasks[u]
                        and stream_tasks[u][0][0] == "valu"
                        and len(bundle.get("valu", [])) < SLOT_LIMITS["valu"]
                    ):
                        if try_add("valu", stream_tasks[u][0][1]):
                            stream_tasks[u].popleft()
                    if stream_tasks[u] and stream_tasks[u][0][0] == "valu":
                        new_active.append(u)
                    else:
                        if stream_tasks[u]:
                            next_head = stream_tasks[u][0][0]
                            if next_head == "load":
                                load_queue.append(u)
                            elif next_head == "valu":
                                valu_queue.append(u)
                            elif next_head == "flow":
                                flow_queue.append(u)
                            elif next_head == "alu":
                                alu_queue.append(u)
                            else:
                                raise RuntimeError(f"Unknown head op {next_head}")

                # If we still have valu capacity, pull more streams this cycle.
                pending = len(valu_queue)
                while (
                    len(bundle.get("valu", [])) < SLOT_LIMITS["valu"]
                    and valu_queue
                    and pending > 0
                ):
                    u = valu_queue.popleft()
                    pending -= 1
                    if (
                        stream_tasks[u]
                        and stream_tasks[u][0][0] == "valu"
                        and len(bundle.get("valu", [])) < SLOT_LIMITS["valu"]
                    ):
                        if try_add("valu", stream_tasks[u][0][1]):
                            stream_tasks[u].popleft()
                    if stream_tasks[u] and stream_tasks[u][0][0] == "valu":
                        new_active.append(u)
                    else:
                        if stream_tasks[u]:
                            next_head = stream_tasks[u][0][0]
                            if next_head == "load":
                                load_queue.append(u)
                            elif next_head == "valu":
                                valu_queue.append(u)
                            elif next_head == "flow":
                                flow_queue.append(u)
                            elif next_head == "alu":
                                alu_queue.append(u)
                            else:
                                raise RuntimeError(f"Unknown head op {next_head}")

                active_valu[:] = new_active
                active_alu[:] = new_active_alu
                refill_active()
                refill_active_alu()

                if not bundle:
                    raise RuntimeError("Scheduler produced an empty bundle")

                self.instrs.append(bundle)

        for block_start in range(0, total_vec_count, vec_count):
            block_n = min(vec_count, total_vec_count - block_start)
            compute_addr_vals(block_start, block_n)
            schedule_streams(build_stream_tasks(block_n))
            for u in range(block_n):
                self.add("store", ("vstore", addr_vals[u], v_val[u]))

        for i in range(vec_batch, per_core_batch):
            i_const = self.scratch_const(i)
            self.add("alu", ("+", tmp_addr_val, self.scratch["inp_values_p"], i_const))
            self.add("load", ("load", tmp_val, tmp_addr_val))
            self.add("alu", ("+", tmp_idx, zero_const, zero_const))

            for r in range(rounds):
                root_round = (r % rounds_period) == 0
                if root_round:
                    self.add("alu", ("^", tmp_val, tmp_val, root_node))
                else:
                    self.add(
                        "alu",
                        ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx),
                    )
                    self.add("load", ("load", tmp_node_val, tmp_addr))
                    self.add("alu", ("^", tmp_val, tmp_val, tmp_node_val))
                for engine, slot in self.build_hash(tmp_val, tmp1, tmp2):
                    self.add(engine, slot)
                self.add("alu", ("&", tmp1, tmp_val, one_const))
                self.add("alu", ("+", tmp1, tmp1, one_const))
                self.add("alu", ("*", tmp_idx, tmp_idx, two_const))
                self.add("alu", ("+", tmp_idx, tmp_idx, tmp1))
                self.add("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]))
                self.add("alu", ("*", tmp_idx, tmp_idx, tmp1))

            self.add("store", ("store", tmp_addr_val, tmp_val))

        # Required to match with the yield in reference_kernel2.
        self.add("flow", ("pause",))
        self.flush_slots()

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
    print(kb.debug_info())
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
