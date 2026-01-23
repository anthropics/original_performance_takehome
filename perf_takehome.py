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
    def __init__(
        self,
        unroll: int = 6,
        load_interleave: int = 2,
        compute_schedule: str = "stagewise",
        hash_interleave: tuple[int, int, int] | None = None,
        block_group: int | None = None,
        gather_strategy: str = "by_buffer",
    ):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.unroll = unroll
        self.load_interleave = load_interleave
        self.compute_schedule = compute_schedule
        self.hash_interleave = hash_interleave
        self.block_group = block_group
        self.gather_strategy = gather_strategy

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            # Simple slot packing that just uses one slot per instruction bundle
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        instrs: list[Instruction] = []
        current: dict[Engine, list[tuple]] = {}
        current_reads: set[int] = set()
        current_writes: set[int] = set()
        engine_counts = defaultdict(int)

        def flush():
            nonlocal current, current_reads, current_writes, engine_counts
            if current:
                instrs.append(current)
            current = {}
            current_reads = set()
            current_writes = set()
            engine_counts = defaultdict(int)

        for engine, slot in slots:
            if engine == "debug":
                flush()
                instrs.append({engine: [slot]})
                continue

            if self._is_barrier(engine, slot):
                flush()
                instrs.append({engine: [slot]})
                continue

            reads, writes = self._slot_access(engine, slot)
            if engine_counts[engine] >= SLOT_LIMITS[engine] or reads & current_writes:
                flush()

            current.setdefault(engine, []).append(slot)
            engine_counts[engine] += 1
            current_reads.update(reads)
            current_writes.update(writes)

        flush()
        return instrs

    def _vec_range(self, base: int) -> set[int]:
        return set(range(base, base + VLEN))

    def _slot_access(self, engine: Engine, slot: tuple) -> tuple[set[int], set[int]]:
        reads: set[int] = set()
        writes: set[int] = set()

        if engine == "alu":
            _, dest, a1, a2 = slot
            reads.update([a1, a2])
            writes.add(dest)
        elif engine == "valu":
            match slot:
                case ("vbroadcast", dest, src):
                    reads.add(src)
                    writes.update(self._vec_range(dest))
                case ("multiply_add", dest, a, b, c):
                    reads.update(self._vec_range(a))
                    reads.update(self._vec_range(b))
                    reads.update(self._vec_range(c))
                    writes.update(self._vec_range(dest))
                case (op, dest, a1, a2):
                    reads.update(self._vec_range(a1))
                    reads.update(self._vec_range(a2))
                    writes.update(self._vec_range(dest))
        elif engine == "load":
            match slot:
                case ("load", dest, addr):
                    reads.add(addr)
                    writes.add(dest)
                case ("load_offset", dest, addr, offset):
                    reads.add(addr + offset)
                    writes.add(dest + offset)
                case ("vload", dest, addr):
                    reads.add(addr)
                    writes.update(self._vec_range(dest))
                case ("const", dest, _val):
                    writes.add(dest)
        elif engine == "store":
            match slot:
                case ("store", addr, src):
                    reads.update([addr, src])
                case ("vstore", addr, src):
                    reads.add(addr)
                    reads.update(self._vec_range(src))
        elif engine == "flow":
            match slot:
                case ("select", dest, cond, a, b):
                    reads.update([cond, a, b])
                    writes.add(dest)
                case ("add_imm", dest, a, _imm):
                    reads.add(a)
                    writes.add(dest)
                case ("vselect", dest, cond, a, b):
                    reads.update(self._vec_range(cond))
                    reads.update(self._vec_range(a))
                    reads.update(self._vec_range(b))
                    writes.update(self._vec_range(dest))
                case ("trace_write", val):
                    reads.add(val)
                case ("cond_jump", cond, addr):
                    reads.update([cond, addr])
                case ("cond_jump_rel", cond, _offset):
                    reads.add(cond)
                case ("jump", addr):
                    reads.add(addr)
                case ("jump_indirect", addr):
                    reads.add(addr)
                case ("coreid", dest):
                    writes.add(dest)
        return reads, writes

    def _is_barrier(self, engine: Engine, slot: tuple) -> bool:
        if engine != "flow":
            return False
        return slot[0] in {"pause", "halt", "jump", "jump_indirect", "cond_jump", "cond_jump_rel"}

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

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_scratch(name, VLEN)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.vconst_map[val] = vec_addr
        return self.vconst_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i, debug: bool = False):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            if debug:
                slots.append(
                    ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
                )

        return slots

    def build_hash_vec(
        self, val_hash_addr, tmp1, tmp2, round, base_i, lanes, debug: bool = False
    ):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(
                ("valu", (op1, tmp1, val_hash_addr, self.scratch_vconst(val1)))
            )
            slots.append(
                ("valu", (op3, tmp2, val_hash_addr, self.scratch_vconst(val3)))
            )
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            if debug and round is not None:
                keys = [
                    (round, base_i + lane, "hash_stage", hi) for lane in range(lanes)
                ]
                slots.append(("debug", ("vcompare", val_hash_addr, keys)))

        return slots

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        debug: bool = False,
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized implementation using SIMD lanes for most arithmetic.
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

        zero_v = self.scratch_vconst(0, "zero_v")
        one_v = self.scratch_vconst(1, "one_v")
        two_v = self.scratch_vconst(2, "two_v")

        forest_values_p_v = self.alloc_scratch("forest_values_p_v", VLEN)
        self.add("valu", ("vbroadcast", forest_values_p_v, self.scratch["forest_values_p"]))
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        self.add("valu", ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        if debug:
            self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scratch arrays for indices/values (kept in scratch across rounds)
        idx_base = self.alloc_scratch("idx_buf", batch_size)
        val_base = self.alloc_scratch("val_buf", batch_size)

        # Vector scratch registers
        vec_idx = self.alloc_scratch("vec_idx", VLEN)
        vec_val = self.alloc_scratch("vec_val", VLEN)
        vec_node_val = self.alloc_scratch("vec_node_val", VLEN)
        vec_addr = self.alloc_scratch("vec_addr", VLEN)
        vec_tmp1 = self.alloc_scratch("vec_tmp1", VLEN)
        vec_tmp2 = self.alloc_scratch("vec_tmp2", VLEN)
        vec_tmp3 = self.alloc_scratch("vec_tmp3", VLEN)
        vec_tmp1_b = self.alloc_scratch("vec_tmp1_b", VLEN)
        vec_tmp2_b = self.alloc_scratch("vec_tmp2_b", VLEN)
        vec_tmp3_b = self.alloc_scratch("vec_tmp3_b", VLEN)
        vec_node_val_b = self.alloc_scratch("vec_node_val_b", VLEN)
        vec_addr_b = self.alloc_scratch("vec_addr_b", VLEN)

        # Scalar scratch registers (tail handling)
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr_idx = self.alloc_scratch("tmp_addr_idx")
        tmp_addr_val = self.alloc_scratch("tmp_addr_val")
        tmp_addr_idx_b = self.alloc_scratch("tmp_addr_idx_b")
        tmp_addr_val_b = self.alloc_scratch("tmp_addr_val_b")
        tmp_addr = self.alloc_scratch("tmp_addr")

        def debug_slot(slot):
            if debug:
                body.append(("debug", slot))

        vec_batch = batch_size - (batch_size % VLEN)

        unroll = self.unroll
        load_interleave = self.load_interleave
        compute_schedule = self.compute_schedule.lower()
        gather_strategy = self.gather_strategy.lower()
        if gather_strategy not in {"by_buffer", "round_robin"}:
            gather_strategy = "by_buffer"
        hash_interleave = self.hash_interleave or (
            load_interleave,
            load_interleave,
            load_interleave,
        )
        block_group = self.block_group if self.block_group and self.block_group > 0 else unroll
        tmp_sets = [
            (vec_tmp1, vec_tmp2, vec_tmp3),
            (vec_tmp1_b, vec_tmp2_b, vec_tmp3_b),
        ]
        for ti in range(2, unroll):
            tmp_sets.append(
                (
                    self.alloc_scratch(f"vec_tmp1_{ti}", VLEN),
                    self.alloc_scratch(f"vec_tmp2_{ti}", VLEN),
                    self.alloc_scratch(f"vec_tmp3_{ti}", VLEN),
                )
            )

        buf_a = [
            {
                "node": vec_node_val,
                "addr": vec_addr,
            }
        ]
        buf_b = [
            {
                "node": vec_node_val_b,
                "addr": vec_addr_b,
            }
        ]
        for bi in range(1, unroll):
            buf_a.append(
                {
                    "node": self.alloc_scratch(f"vec_node_val_a{bi}", VLEN),
                    "addr": self.alloc_scratch(f"vec_addr_a{bi}", VLEN),
                }
            )
            buf_b.append(
                {
                    "node": self.alloc_scratch(f"vec_node_val_b{bi}", VLEN),
                    "addr": self.alloc_scratch(f"vec_addr_b{bi}", VLEN),
                }
            )

        def prefetch(buf, idx_addr):
            body.append(("valu", ("+", buf["addr"], idx_addr, forest_values_p_v)))

        def gather_list(buf):
            return [
                ("load", ("load_offset", buf["node"], buf["addr"], lane))
                for lane in range(VLEN)
            ]

        def gather_list_round_robin(buffers):
            return [
                ("load", ("load_offset", buf["node"], buf["addr"], lane))
                for lane in range(VLEN)
                for buf in buffers
            ]

        def maybe_emit_loads(extra_loads, count: int | None = None):
            if count is None:
                count = load_interleave
            for _ in range(count):
                if not extra_loads:
                    break
                body.append(extra_loads.pop(0))

        def emit_hash_vec_interleaved(
            val_addr, tmp1_addr, tmp2_addr, round_idx, base_i, lanes, extra_loads
        ):
            h1, h2, h3 = hash_interleave
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                body.append(
                    ("valu", (op1, tmp1_addr, val_addr, self.scratch_vconst(val1)))
                )
                maybe_emit_loads(extra_loads, h1)
                body.append(
                    ("valu", (op3, tmp2_addr, val_addr, self.scratch_vconst(val3)))
                )
                maybe_emit_loads(extra_loads, h2)
                body.append(("valu", (op2, val_addr, tmp1_addr, tmp2_addr)))
                maybe_emit_loads(extra_loads, h3)
                if debug:
                    keys = [
                        (round_idx, base_i + lane, "hash_stage", hi)
                        for lane in range(lanes)
                    ]
                    body.append(("debug", ("vcompare", val_addr, keys)))

        def emit_compute(idx_addr, val_addr, node_addr, round_idx, base_i, extra_loads):
            body.append(("valu", ("^", val_addr, val_addr, node_addr)))
            emit_hash_vec_interleaved(
                val_addr, vec_tmp1, vec_tmp2, round_idx, base_i, VLEN, extra_loads
            )
            body.append(("valu", ("&", vec_tmp1, val_addr, one_v)))
            maybe_emit_loads(extra_loads)
            body.append(("valu", ("+", vec_tmp3, vec_tmp1, one_v)))
            maybe_emit_loads(extra_loads)
            body.append(("valu", ("multiply_add", idx_addr, idx_addr, two_v, vec_tmp3)))
            maybe_emit_loads(extra_loads)
            body.append(("valu", ("<", vec_tmp1, idx_addr, n_nodes_v)))
            maybe_emit_loads(extra_loads)
            body.append(("valu", ("multiply_add", idx_addr, idx_addr, vec_tmp1, zero_v)))
            if extra_loads:
                body.extend(extra_loads)
                extra_loads.clear()

        def emit_compute_blockwise(blocks, extra_loads):
            for block in blocks:
                body.append(("valu", ("^", block["val"], block["val"], block["node"])))
                emit_hash_vec_interleaved(
                    block["val"],
                    block["tmp1"],
                    block["tmp2"],
                    None,
                    0,
                    VLEN,
                    extra_loads,
                )
                body.append(("valu", ("&", block["tmp1"], block["val"], one_v)))
                maybe_emit_loads(extra_loads)
                body.append(("valu", ("+", block["tmp3"], block["tmp1"], one_v)))
                maybe_emit_loads(extra_loads)
                body.append(
                    (
                        "valu",
                        ("multiply_add", block["idx"], block["idx"], two_v, block["tmp3"]),
                    )
                )
                maybe_emit_loads(extra_loads)
                body.append(("valu", ("<", block["tmp1"], block["idx"], n_nodes_v)))
                maybe_emit_loads(extra_loads)
                body.append(
                    (
                        "valu",
                        ("multiply_add", block["idx"], block["idx"], block["tmp1"], zero_v),
                    )
                )
            if extra_loads:
                body.extend(extra_loads)
                extra_loads.clear()

        def iter_block_groups(blocks):
            if block_group >= len(blocks):
                yield blocks
                return
            for gi in range(0, len(blocks), block_group):
                yield blocks[gi : gi + block_group]

        def emit_compute_interleaved(blocks, extra_loads):
            def drain_loads(count: int = load_interleave):
                maybe_emit_loads(extra_loads, count)
            for group in iter_block_groups(blocks):
                for block in group:
                    body.append(("valu", ("^", block["val"], block["val"], block["node"])))
                drain_loads()

                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    for block in group:
                        body.append(
                            (
                                "valu",
                                (op1, block["tmp1"], block["val"], self.scratch_vconst(val1)),
                            )
                        )
                    drain_loads()
                    for block in group:
                        body.append(
                            (
                                "valu",
                                (op3, block["tmp2"], block["val"], self.scratch_vconst(val3)),
                            )
                        )
                    drain_loads()
                    for block in group:
                        body.append(
                            ("valu", (op2, block["val"], block["tmp1"], block["tmp2"]))
                        )
                    drain_loads()

                for block in group:
                    body.append(("valu", ("&", block["tmp1"], block["val"], one_v)))
                drain_loads()
                for block in group:
                    body.append(("valu", ("+", block["tmp3"], block["tmp1"], one_v)))
                drain_loads()
                for block in group:
                    body.append(
                        (
                            "valu",
                            ("multiply_add", block["idx"], block["idx"], two_v, block["tmp3"]),
                        )
                    )
                drain_loads()
                for block in group:
                    body.append(("valu", ("<", block["tmp1"], block["idx"], n_nodes_v)))
                drain_loads()
                for block in group:
                    body.append(
                        (
                            "valu",
                            ("multiply_add", block["idx"], block["idx"], block["tmp1"], zero_v),
                        )
                    )
                if extra_loads:
                    body.extend(extra_loads)
                    extra_loads.clear()

        if not debug and vec_batch:
            for i in range(0, vec_batch, VLEN):
                i_const = self.scratch_const(i)
                idx_addr = idx_base + i
                val_addr = val_base + i
                body.append(("alu", ("+", tmp_addr_idx, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("vload", idx_addr, tmp_addr_idx)))
                body.append(("alu", ("+", tmp_addr_val, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("vload", val_addr, tmp_addr_val)))

        for round in range(rounds):
            if debug:
                for i in range(0, vec_batch, VLEN):
                    i_const = self.scratch_const(i)
                    lane_keys = [(round, i + lane, "idx") for lane in range(VLEN)]
                    # idx = mem[inp_indices_p + i : i+VLEN]
                    body.append(
                        ("alu", ("+", tmp_addr_idx, self.scratch["inp_indices_p"], i_const))
                    )
                    body.append(
                        ("alu", ("+", tmp_addr_val, self.scratch["inp_values_p"], i_const))
                    )
                    body.append(("load", ("vload", vec_idx, tmp_addr_idx)))
                    debug_slot(("vcompare", vec_idx, lane_keys))
                    # val = mem[inp_values_p + i : i+VLEN]
                    body.append(("load", ("vload", vec_val, tmp_addr_val)))
                    debug_slot(
                        ("vcompare", vec_val, [(round, i + lane, "val") for lane in range(VLEN)])
                    )
                    # node_val = mem[forest_values_p + idx]
                    body.append(("valu", ("+", vec_addr, vec_idx, forest_values_p_v)))
                    for lane in range(VLEN):
                        body.append(("load", ("load_offset", vec_node_val, vec_addr, lane)))
                    debug_slot(
                        (
                            "vcompare",
                            vec_node_val,
                            [(round, i + lane, "node_val") for lane in range(VLEN)],
                        )
                    )
                    # val = myhash(val ^ node_val)
                    body.append(("valu", ("^", vec_val, vec_val, vec_node_val)))
                    body.extend(
                        self.build_hash_vec(
                            vec_val, vec_tmp1, vec_tmp2, round, i, VLEN, debug=debug
                        )
                    )
                    debug_slot(
                        (
                            "vcompare",
                            vec_val,
                            [(round, i + lane, "hashed_val") for lane in range(VLEN)],
                        )
                    )
                    # idx = 2*idx + (1 + (val & 1))
                    body.append(("valu", ("&", vec_tmp1, vec_val, one_v)))
                    body.append(("valu", ("+", vec_tmp3, vec_tmp1, one_v)))
                    body.append(("valu", ("multiply_add", vec_idx, vec_idx, two_v, vec_tmp3)))
                    debug_slot(
                        ("vcompare", vec_idx, [(round, i + lane, "next_idx") for lane in range(VLEN)])
                    )
                    # idx = 0 if idx >= n_nodes else idx
                    body.append(("valu", ("<", vec_tmp1, vec_idx, n_nodes_v)))
                    body.append(("flow", ("vselect", vec_idx, vec_tmp1, vec_idx, zero_v)))
                    debug_slot(
                        (
                            "vcompare",
                            vec_idx,
                            [(round, i + lane, "wrapped_idx") for lane in range(VLEN)],
                        )
                    )
                    # mem[inp_indices_p + i] = idx
                    body.append(("store", ("vstore", tmp_addr_idx, vec_idx)))
                    # mem[inp_values_p + i] = val
                    body.append(("store", ("vstore", tmp_addr_val, vec_val)))
            else:
                if vec_batch:
                    group_stride = unroll * VLEN
                    i = 0
                    cur_buffers = buf_a
                    next_buffers = buf_b
                    cur_blocks = []
                    cur_gather_buffers = []
                    for u in range(unroll):
                        base = i + u * VLEN
                        if base >= vec_batch:
                            break
                        idx_addr = idx_base + base
                        val_addr = val_base + base
                        buf = cur_buffers[u]
                        tmp1, tmp2, tmp3 = tmp_sets[u]
                        prefetch(buf, idx_addr)
                        if gather_strategy == "round_robin":
                            cur_gather_buffers.append(buf)
                        else:
                            body.extend(gather_list(buf))
                        cur_blocks.append(
                            {
                                "idx": idx_addr,
                                "val": val_addr,
                                "node": buf["node"],
                                "tmp1": tmp1,
                                "tmp2": tmp2,
                                "tmp3": tmp3,
                            }
                        )
                    if gather_strategy == "round_robin" and cur_gather_buffers:
                        body.extend(gather_list_round_robin(cur_gather_buffers))
                    i += group_stride
                    while cur_blocks:
                        extra_loads = []
                        next_blocks = []
                        next_gather_buffers = []
                        if i < vec_batch:
                            for u in range(unroll):
                                base = i + u * VLEN
                                if base >= vec_batch:
                                    break
                                idx_addr = idx_base + base
                                val_addr = val_base + base
                                buf = next_buffers[u]
                                tmp1, tmp2, tmp3 = tmp_sets[u]
                                prefetch(buf, idx_addr)
                                if gather_strategy == "round_robin":
                                    next_gather_buffers.append(buf)
                                else:
                                    extra_loads.extend(gather_list(buf))
                                next_blocks.append(
                                    {
                                        "idx": idx_addr,
                                        "val": val_addr,
                                        "node": buf["node"],
                                        "tmp1": tmp1,
                                        "tmp2": tmp2,
                                        "tmp3": tmp3,
                                    }
                                )
                            if gather_strategy == "round_robin" and next_gather_buffers:
                                extra_loads.extend(gather_list_round_robin(next_gather_buffers))
                        if compute_schedule == "blockwise":
                            emit_compute_blockwise(cur_blocks, extra_loads)
                        else:
                            emit_compute_interleaved(cur_blocks, extra_loads)
                        cur_blocks = next_blocks
                        cur_buffers, next_buffers = next_buffers, cur_buffers
                        i += group_stride

            for i in range(vec_batch, batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                debug_slot(("compare", tmp_idx, (round, i, "idx")))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                debug_slot(("compare", tmp_val, (round, i, "val")))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                debug_slot(("compare", tmp_node_val, (round, i, "node_val")))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i, debug=debug))
                debug_slot(("compare", tmp_val, (round, i, "hashed_val")))
                # idx = 2*idx + (1 + (val & 1))
                body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                body.append(("alu", ("+", tmp3, tmp1, one_const)))
                body.append(("alu", ("<<", tmp_idx, tmp_idx, one_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                debug_slot(("compare", tmp_idx, (round, i, "next_idx")))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                debug_slot(("compare", tmp_idx, (round, i, "wrapped_idx")))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        if not debug and vec_batch:
            for i in range(0, vec_batch, VLEN):
                i_const = self.scratch_const(i)
                idx_addr = idx_base + i
                val_addr = val_base + i
                body.append(("alu", ("+", tmp_addr_idx, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr_idx, idx_addr)))
                body.append(("alu", ("+", tmp_addr_val, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr_val, val_addr)))

        body_instrs = self.build(body, vliw=True)
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
