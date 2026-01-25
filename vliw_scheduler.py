"""
Greedy list scheduler for VLIW instruction bundling.

This module is intentionally conservative: it preserves program order across
any shared scratch address, and orders memory ops by address register to avoid
obvious aliasing hazards. It is meant as a safe starting point for faster
bundling, not the final performance solution.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from problem import SCRATCH_SIZE, VLEN

# Slot is (engine, payload)
Slot = Tuple[str, Tuple]


@dataclass(frozen=True)
class SlotDeps:
    reads: set[int]
    writes: set[int]
    is_mem: bool
    mem_addr_reg: int | None
    is_store: bool


def _slot_deps(
    engine: str, slot: Tuple, *, load_offset_reads_base: bool = False
) -> SlotDeps:
    """
    Return read/write scratch dependencies for a slot.

    This is based on the simulator ISA conventions:
    - For ALU/VALU ops: (op, dest, a1, a2) => reads a1, a2; writes dest.
    - For load: ("load", dest, addr) => reads addr; writes dest.
    - For load const: ("const", dest, value) => writes dest.
    - For store: ("store", addr, src) => reads addr, src; writes memory.
    - For flow select: ("select", dest, cond, a, b) => reads cond, a, b; writes dest.
    - For flow jump: ("jump", target) => no scratch deps (control-only).
    - For debug: ignored by scheduler (no deps).

    This is conservative and does not model memory aliasing precisely.
    """
    if engine == "debug":
        # Debug compares read scratch; treat as reads to enforce ordering.
        if slot[0] == "compare":
            _, loc, _key = slot
            return SlotDeps(
                reads={loc},
                writes=set(),
                is_mem=False,
                mem_addr_reg=None,
                is_store=False,
            )
        if slot[0] == "vcompare":
            _, loc, keys = slot
            return SlotDeps(
                reads=set(range(loc, loc + len(keys))),
                writes=set(),
                is_mem=False,
                mem_addr_reg=None,
                is_store=False,
            )
        return SlotDeps(set(), set(), False, None, False)

    op = slot[0]
    reads: set[int] = set()
    writes: set[int] = set()
    is_mem = False
    mem_addr_reg = None
    is_store = False

    if engine == "alu":
        # (op, dest, a1, a2)
        if op in ("~",):  # unary ops if any
            _, dest, a1 = slot
            reads.add(a1)
            writes.add(dest)
        else:
            _, dest, a1, a2 = slot
            reads.update((a1, a2))
            writes.add(dest)
    elif engine == "valu":
        # Vector ops: treat ranges of VLEN scratch addresses.
        if op == "vbroadcast":
            _, dest, src = slot
            reads.add(src)
            writes.update(range(dest, dest + VLEN))
        elif op == "multiply_add":
            _, dest, a, b, c = slot
            reads.update(range(a, a + VLEN))
            reads.update(range(b, b + VLEN))
            reads.update(range(c, c + VLEN))
            writes.update(range(dest, dest + VLEN))
        else:
            _, dest, a1, a2 = slot
            reads.update(range(a1, a1 + VLEN))
            reads.update(range(a2, a2 + VLEN))
            writes.update(range(dest, dest + VLEN))
    elif engine == "load":
        if op == "const":
            _, dest, _val = slot
            writes.add(dest)
        elif op == "load":
            _, dest, addr = slot
            reads.add(addr)
            writes.add(dest)
            is_mem = True
            is_store = False
            mem_addr_reg = addr
        elif op == "load_offset":
            _, dest, addr, offset = slot
            if load_offset_reads_base:
                reads.add(addr)
            else:
                reads.add(addr + offset)
            writes.add(dest)
            is_mem = True
            is_store = False
            # Keep memory keying based on base+offset form for safety.
            mem_addr_reg = addr + offset
        elif op == "vload":
            # ("vload", dest_vec, addr)
            reads.add(slot[2])
            writes.update(range(slot[1], slot[1] + VLEN))
            is_mem = True
            is_store = False
            mem_addr_reg = slot[2]
        elif op == "vbroadcast":
            _, dest_vec, src = slot
            reads.add(src)
            writes.update(range(dest_vec, dest_vec + VLEN))
        else:
            # Unknown load op; be conservative.
            for v in slot[1:]:
                if isinstance(v, int):
                    reads.add(v)
            is_mem = True
    elif engine == "store":
        if op == "store":
            _, addr, src = slot
            reads.update((addr, src))
            is_mem = True
            is_store = True
            mem_addr_reg = addr
        elif op == "vstore":
            # ("vstore", addr, src_vec)
            reads.add(slot[1])
            reads.update(range(slot[2], slot[2] + VLEN))
            is_mem = True
            is_store = True
            mem_addr_reg = slot[1]
        else:
            for v in slot[1:]:
                if isinstance(v, int):
                    reads.add(v)
            is_mem = True
    elif engine == "flow":
        if op == "select":
            _, dest, cond, a, b = slot
            reads.update((cond, a, b))
            writes.add(dest)
        elif op in ("pause", "halt"):
            pass
        elif op in ("jump", "jump_if"):
            # These are control flow; depend on condition if present.
            if op == "jump_if":
                _, cond, _target = slot
                reads.add(cond)
        elif op == "vselect":
            _, dest, cond, a, b = slot
            reads.update(range(cond, cond + VLEN))
            reads.update(range(a, a + VLEN))
            reads.update(range(b, b + VLEN))
            writes.update(range(dest, dest + VLEN))
        else:
            # Unknown flow op; be conservative.
            for v in slot[1:]:
                if isinstance(v, int):
                    reads.add(v)
    else:
        # Unknown engine; be conservative.
        for v in slot[1:]:
            if isinstance(v, int):
                reads.add(v)

    return SlotDeps(
        reads=reads,
        writes=writes,
        is_mem=is_mem,
        mem_addr_reg=mem_addr_reg,
        is_store=is_store,
    )


def schedule_slots(
    slots: Sequence[Slot],
    slot_limits: Dict[str, int],
    *,
    serialize_mem: bool = False,
    window: int = 256,
    disambiguate_mem: bool = False,
    load_offset_reads_base: bool = False,
    weighted_priority: bool = False,
    slack_tie_break: bool = False,
    priority_weights: Dict[str, int] | None = None,
    global_pick: bool = False,
    bundle_repair: bool = False,
    repair_weight: int = 1,
    rename_war_waw: bool = False,
    rename_vectors: bool = False,
    debug_stats: bool = False,
) -> List[Dict[str, List[Tuple]]]:
    """
    Schedule a sequence of slots into VLIW bundles.

    This uses a greedy list scheduler with conservative dependency tracking.
    Priority is based on longest-path criticality in the dependency graph.
    Set disambiguate_mem=True to use affine address keys for memory ordering.
    Set load_offset_reads_base=True if load_offset uses base+imm addressing.
    weighted_priority weights the critical path by engine type.
    slack_tie_break prefers low-slack nodes when priorities tie.
    global_pick selects the highest-priority ready ops across engines first.
    bundle_repair tries a one-bundle lookahead swap using successor-unlock score.
    rename_war_waw enables a conservative scratch renamer to reduce WAR/WAW edges.
    rename_vectors enables vector register renaming for full-vector temps.
    debug_stats prints dependency graph and renamer statistics.
    Returns a list of instruction bundles (dict of engine -> list of slots).
    """
    if rename_war_waw:
        scratch_size = SCRATCH_SIZE
        renamed_operands = 0
        renamed_regs: set[int] = set()

        def count_change(old: int, new: int) -> None:
            nonlocal renamed_operands
            if old != new:
                renamed_operands += 1
                renamed_regs.add(old)

        def _slot_rw_indices(engine: str, slot: Tuple) -> tuple[set[int], set[int]]:
            op = slot[0]
            reads: set[int] = set()
            writes: set[int] = set()
            if engine == "debug":
                if op == "compare":
                    _, loc, _key = slot
                    reads.add(loc)
                elif op == "vcompare":
                    _, loc, keys = slot
                    reads.update(range(loc, loc + len(keys)))
                return reads, writes
            if engine == "alu":
                if op in ("~",):
                    _, dest, a1 = slot
                    reads.add(a1)
                    writes.add(dest)
                else:
                    _, dest, a1, a2 = slot
                    reads.update((a1, a2))
                    writes.add(dest)
            elif engine == "valu":
                if op == "vbroadcast":
                    _, dest, src = slot
                    reads.add(src)
                    writes.update(range(dest, dest + VLEN))
                elif op == "multiply_add":
                    _, dest, a, b, c = slot
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                    reads.update(range(c, c + VLEN))
                    writes.update(range(dest, dest + VLEN))
                else:
                    _, dest, a1, a2 = slot
                    reads.update(range(a1, a1 + VLEN))
                    reads.update(range(a2, a2 + VLEN))
                    writes.update(range(dest, dest + VLEN))
            elif engine == "load":
                if op == "const":
                    _, dest, _val = slot
                    writes.add(dest)
                elif op == "load":
                    _, dest, addr = slot
                    reads.add(addr)
                    writes.add(dest)
                elif op == "load_offset":
                    _, dest, addr, offset = slot
                    if load_offset_reads_base:
                        reads.add(addr)
                    else:
                        reads.add(addr + offset)
                    writes.add(dest + offset)
                elif op == "vload":
                    reads.add(slot[2])
                    writes.update(range(slot[1], slot[1] + VLEN))
                elif op == "vbroadcast":
                    _, dest_vec, src = slot
                    reads.add(src)
                    writes.update(range(dest_vec, dest_vec + VLEN))
                else:
                    for v in slot[1:]:
                        if isinstance(v, int):
                            reads.add(v)
            elif engine == "store":
                if op == "store":
                    _, addr, src = slot
                    reads.update((addr, src))
                elif op == "vstore":
                    reads.add(slot[1])
                    reads.update(range(slot[2], slot[2] + VLEN))
                else:
                    for v in slot[1:]:
                        if isinstance(v, int):
                            reads.add(v)
            elif engine == "flow":
                if op == "select":
                    _, dest, cond, a, b = slot
                    reads.update((cond, a, b))
                    writes.add(dest)
                elif op == "add_imm":
                    _, dest, a, _imm = slot
                    reads.add(a)
                    writes.add(dest)
                elif op == "jump_if":
                    _, cond, _target = slot
                    reads.add(cond)
                elif op == "vselect":
                    _, dest, cond, a, b = slot
                    reads.update(range(cond, cond + VLEN))
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                    writes.update(range(dest, dest + VLEN))
                else:
                    for v in slot[1:]:
                        if isinstance(v, int):
                            reads.add(v)
            else:
                for v in slot[1:]:
                    if isinstance(v, int):
                        reads.add(v)
            return reads, writes

        vec_width_by_base: Dict[int, int] = {}
        vec_ineligible: set[int] = set()
        addr_regs: set[int] = set()

        def record_vec_base(base: int, width: int) -> None:
            if base in vec_width_by_base and vec_width_by_base[base] != width:
                vec_ineligible.add(base)
            else:
                vec_width_by_base.setdefault(base, width)

        for engine, slot in slots:
            op = slot[0]
            if engine == "valu":
                if op == "vbroadcast":
                    record_vec_base(slot[1], VLEN)
                elif op == "multiply_add":
                    record_vec_base(slot[1], VLEN)
                    record_vec_base(slot[2], VLEN)
                    record_vec_base(slot[3], VLEN)
                    record_vec_base(slot[4], VLEN)
                else:
                    record_vec_base(slot[1], VLEN)
                    record_vec_base(slot[2], VLEN)
                    record_vec_base(slot[3], VLEN)
            elif engine == "load":
                if op == "vload":
                    record_vec_base(slot[1], VLEN)
                elif op == "vbroadcast":
                    record_vec_base(slot[1], VLEN)
                elif op == "load_offset":
                    record_vec_base(slot[1], VLEN)
                    record_vec_base(slot[2], VLEN)
            elif engine == "store":
                if op == "vstore":
                    record_vec_base(slot[2], VLEN)
            elif engine == "flow":
                if op == "vselect":
                    record_vec_base(slot[1], VLEN)
                    record_vec_base(slot[2], VLEN)
                    record_vec_base(slot[3], VLEN)
                    record_vec_base(slot[4], VLEN)
            elif engine == "debug" and op == "vcompare":
                record_vec_base(slot[1], len(slot[2]))

            # Track memory address regs to keep them fixed.
            if engine == "load":
                if op == "load":
                    addr_regs.add(slot[2])
                elif op == "load_offset":
                    _, _dest, addr, offset = slot
                    addr_regs.add(addr + offset)
                elif op == "vload":
                    addr_regs.add(slot[2])
            elif engine == "store":
                if op == "store":
                    addr_regs.add(slot[1])
                elif op == "vstore":
                    addr_regs.add(slot[1])

        vec_base_for_reg_any: Dict[int, int] = {}
        for base, width in vec_width_by_base.items():
            for r in range(base, base + width):
                if r in vec_base_for_reg_any and vec_base_for_reg_any[r] != base:
                    vec_ineligible.add(base)
                    vec_ineligible.add(vec_base_for_reg_any[r])
                else:
                    vec_base_for_reg_any[r] = base

        per_slot_rw: List[tuple[set[int], set[int]]] = []
        vec_partial_access: set[int] = set()
        for engine, slot in slots:
            reads, writes = _slot_rw_indices(engine, slot)
            per_slot_rw.append((reads, writes))
            for r in reads:
                base = vec_base_for_reg_any.get(r)
                if base is None:
                    continue
                width = vec_width_by_base.get(base, VLEN)
                if width != VLEN:
                    vec_ineligible.add(base)
                    continue
                if base in vec_partial_access:
                    continue
                if not all((base + o) in reads for o in range(width)):
                    vec_partial_access.add(base)
            for w in writes:
                base = vec_base_for_reg_any.get(w)
                if base is None:
                    continue
                width = vec_width_by_base.get(base, VLEN)
                if width != VLEN:
                    vec_ineligible.add(base)
                    continue
                if base in vec_partial_access:
                    continue
                if not all((base + o) in writes for o in range(width)):
                    vec_partial_access.add(base)

        vec_eligible: set[int] = set()
        if rename_vectors:
            for base, width in vec_width_by_base.items():
                if width != VLEN:
                    continue
                if base in vec_ineligible or base in vec_partial_access:
                    continue
                if any((base + o) in addr_regs for o in range(width)):
                    continue
                vec_eligible.add(base)

        vec_base_for_reg: Dict[int, int] = {}
        for base in vec_eligible:
            for r in range(base, base + VLEN):
                vec_base_for_reg[r] = base

        blocked_scalar: set[int] = set(addr_regs)
        for base, width in vec_width_by_base.items():
            for r in range(base, base + width):
                blocked_scalar.add(r)

        blocked_fixed: set[int] = set(addr_regs)
        for base, width in vec_width_by_base.items():
            if base not in vec_eligible:
                for r in range(base, base + width):
                    blocked_fixed.add(r)

        scalar_reads: List[List[int]] = []
        scalar_writes: List[List[int]] = []
        scalar_regs: set[int] = set()
        vec_reads: List[List[int]] = []
        vec_writes: List[List[int]] = []
        for reads, writes in per_slot_rw:
            s_reads = [r for r in reads if r not in blocked_scalar]
            s_writes = [w for w in writes if w not in blocked_scalar]
            scalar_reads.append(s_reads)
            scalar_writes.append(s_writes)
            scalar_regs.update(s_reads)
            scalar_regs.update(s_writes)

            v_reads: set[int] = set()
            for r in reads:
                base = vec_base_for_reg.get(r)
                if base is not None:
                    v_reads.add(base)
            v_writes: set[int] = set()
            for w in writes:
                base = vec_base_for_reg.get(w)
                if base is None:
                    continue
                if base in v_writes:
                    continue
                if all((base + o) in writes for o in range(VLEN)):
                    v_writes.add(base)
            vec_reads.append(list(v_reads))
            vec_writes.append(list(v_writes))

        inf = 10**9
        next_read: Dict[int, int] = {r: inf for r in scalar_regs}
        next_write: Dict[int, int] = {r: inf for r in scalar_regs}
        def_last_use: List[Dict[int, int]] = [defaultdict(int) for _ in slots]
        for i in range(len(slots) - 1, -1, -1):
            for w in scalar_writes[i]:
                nr = next_read.get(w, inf)
                nw = next_write.get(w, inf)
                def_last_use[i][w] = nr if nr < nw else i
                next_write[w] = i
            for r in scalar_reads[i]:
                next_read[r] = i

        initial_live_until: Dict[int, int] = {r: -1 for r in scalar_regs}
        seen_write: set[int] = set()
        for i in range(len(slots)):
            for r in scalar_reads[i]:
                if r not in seen_write:
                    initial_live_until[r] = i
            for w in scalar_writes[i]:
                if w not in seen_write:
                    seen_write.add(w)

        next_vec_read: Dict[int, int] = {b: inf for b in vec_eligible}
        next_vec_write: Dict[int, int] = {b: inf for b in vec_eligible}
        def_last_use_vec: List[Dict[int, int]] = [defaultdict(int) for _ in slots]
        for i in range(len(slots) - 1, -1, -1):
            for b in vec_writes[i]:
                nr = next_vec_read.get(b, inf)
                nw = next_vec_write.get(b, inf)
                def_last_use_vec[i][b] = nr if nr < nw else i
                next_vec_write[b] = i
            for b in vec_reads[i]:
                next_vec_read[b] = i

        initial_live_until_vec: Dict[int, int] = {b: -1 for b in vec_eligible}
        seen_vec_write: set[int] = set()
        for i in range(len(slots)):
            for b in vec_reads[i]:
                if b not in seen_vec_write:
                    initial_live_until_vec[b] = i
            for b in vec_writes[i]:
                if b not in seen_vec_write:
                    seen_vec_write.add(b)

        mapping_scalar: Dict[int, int] = {r: r for r in scalar_regs}
        mapping_vec: Dict[int, int] = {b: b for b in vec_eligible}
        vec_phys_owner: Dict[int, int] = {}

        used = [False] * scratch_size
        for r in blocked_fixed:
            if 0 <= r < scratch_size:
                used[r] = True

        phys_live_until: Dict[int, int] = {}
        expire_heap: List[Tuple[int, int]] = []
        for r in scalar_regs:
            if 0 <= r < scratch_size:
                used[r] = True
                phys_live_until[r] = initial_live_until[r]
                expire_heap.append((initial_live_until[r], r))

        vec_live_until: Dict[int, int] = {}
        vec_expire_heap: List[Tuple[int, int]] = []
        for base in vec_eligible:
            phys = mapping_vec[base]
            live_until = initial_live_until_vec.get(base, -1)
            vec_live_until[phys] = live_until
            vec_phys_owner[phys] = base
            for o in range(VLEN):
                if 0 <= phys + o < scratch_size:
                    used[phys + o] = True
            vec_expire_heap.append((live_until, phys))

        import heapq

        heapq.heapify(expire_heap)
        heapq.heapify(vec_expire_heap)

        def free_expired(now: int) -> None:
            while expire_heap and expire_heap[0][0] < now:
                live_until, phys = heapq.heappop(expire_heap)
                if phys_live_until.get(phys) != live_until:
                    continue
                used[phys] = False
                phys_live_until.pop(phys, None)
            while vec_expire_heap and vec_expire_heap[0][0] < now:
                live_until, phys = heapq.heappop(vec_expire_heap)
                if vec_live_until.get(phys) != live_until:
                    continue
                for o in range(VLEN):
                    used[phys + o] = False
                vec_live_until.pop(phys, None)
                owner = vec_phys_owner.pop(phys, None)
                if owner is not None and mapping_vec.get(owner) == phys:
                    mapping_vec[owner] = owner

        def alloc_scalar() -> int | None:
            for reg in range(scratch_size - 1, -1, -1):
                if not used[reg]:
                    used[reg] = True
                    return reg
            return None

        def alloc_vector_block() -> int | None:
            for base in range(scratch_size - VLEN, -1, -1):
                if all(not used[base + o] for o in range(VLEN)):
                    for o in range(VLEN):
                        used[base + o] = True
                    return base
            return None

        new_slots: List[Slot] = []
        for i, (engine, slot) in enumerate(slots):
            free_expired(i)
            op = slot[0]

            def map_r(reg: int) -> int:
                base = vec_base_for_reg.get(reg)
                if base is not None:
                    phys_base = mapping_vec.get(base, base)
                    return phys_base + (reg - base)
                return mapping_scalar.get(reg, reg)

            # Map reads first (before updating writes).
            if engine == "alu":
                if op in ("~",):
                    _, dest, a1 = slot
                    a1_m = map_r(a1)
                    count_change(a1, a1_m)
                else:
                    _, dest, a1, a2 = slot
                    a1_m = map_r(a1)
                    a2_m = map_r(a2)
                    count_change(a1, a1_m)
                    count_change(a2, a2_m)
            elif engine == "valu":
                if op == "vbroadcast":
                    _, dest, src = slot
                    src_m = map_r(src)
                    count_change(src, src_m)
                elif op == "multiply_add":
                    _, dest, a, b, c = slot
                    a_m = map_r(a)
                    b_m = map_r(b)
                    c_m = map_r(c)
                    count_change(a, a_m)
                    count_change(b, b_m)
                    count_change(c, c_m)
                else:
                    _, dest, a1, a2 = slot
                    a1_m = map_r(a1)
                    a2_m = map_r(a2)
                    count_change(a1, a1_m)
                    count_change(a2, a2_m)
            elif engine == "load":
                if op == "load":
                    _, dest, addr = slot
                    addr_m = map_r(addr)
                    count_change(addr, addr_m)
                elif op == "load_offset":
                    _, dest, addr, offset = slot
                    addr_m = map_r(addr)
                    count_change(addr, addr_m)
                elif op == "vload":
                    _, dest, addr = slot
                    addr_m = map_r(addr)
                    count_change(addr, addr_m)
                elif op == "vbroadcast":
                    _, dest, src = slot
                    src_m = map_r(src)
                    count_change(src, src_m)
            elif engine == "store":
                if op == "store":
                    _, addr, src = slot
                    addr_m = map_r(addr)
                    src_m = map_r(src)
                    count_change(addr, addr_m)
                    count_change(src, src_m)
                elif op == "vstore":
                    _, addr, src = slot
                    addr_m = map_r(addr)
                    src_m = map_r(src)
                    count_change(addr, addr_m)
                    count_change(src, src_m)
            elif engine == "flow":
                if op == "select":
                    _, dest, cond, a, b = slot
                    cond_m = map_r(cond)
                    a_m = map_r(a)
                    b_m = map_r(b)
                    count_change(cond, cond_m)
                    count_change(a, a_m)
                    count_change(b, b_m)
                elif op == "add_imm":
                    _, dest, a, imm = slot
                    a_m = map_r(a)
                    count_change(a, a_m)
                elif op == "jump_if":
                    _, cond, target = slot
                    cond_m = map_r(cond)
                    count_change(cond, cond_m)
                elif op == "vselect":
                    _, dest, cond, a, b = slot
                    cond_m = map_r(cond)
                    a_m = map_r(a)
                    b_m = map_r(b)
                    count_change(cond, cond_m)
                    count_change(a, a_m)
                    count_change(b, b_m)
            elif engine == "debug":
                if op == "compare":
                    _, loc, key = slot
                    loc_m = map_r(loc)
                    count_change(loc, loc_m)
                elif op == "vcompare":
                    _, loc, keys = slot
                    loc_m = map_r(loc)
                    count_change(loc, loc_m)

            # Allocate new physical regs for scalar writes.
            for w in scalar_writes[i]:
                phys = alloc_scalar()
                if phys is None:
                    phys = mapping_scalar[w]
                mapping_scalar[w] = phys
                live_until = def_last_use[i].get(w, i)
                phys_live_until[phys] = live_until
                heapq.heappush(expire_heap, (live_until, phys))

            # Allocate new physical blocks for vector writes.
            for b in vec_writes[i]:
                phys = alloc_vector_block()
                if phys is None:
                    phys = mapping_vec.get(b, b)
                if vec_phys_owner.get(phys) not in (None, b):
                    phys = mapping_vec.get(b, b)
                mapping_vec[b] = phys
                vec_phys_owner[phys] = b
                live_until = def_last_use_vec[i].get(b, i)
                vec_live_until[phys] = live_until
                heapq.heappush(vec_expire_heap, (live_until, phys))

            # Rebuild slot with mapped operands.
            if engine == "alu":
                if op in ("~",):
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, a1_m)
                else:
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, a1_m, a2_m)
            elif engine == "valu":
                if op == "vbroadcast":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, src_m)
                elif op == "multiply_add":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, a_m, b_m, c_m)
                else:
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, a1_m, a2_m)
            elif engine == "load":
                if op == "const":
                    _, dest, val = slot
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, val)
                elif op == "load":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, addr_m)
                elif op == "load_offset":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, addr_m, offset)
                elif op == "vload":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, addr_m)
                elif op == "vbroadcast":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, src_m)
                else:
                    new_slot = slot
            elif engine == "store":
                if op == "store":
                    new_slot = (op, addr_m, src_m)
                elif op == "vstore":
                    new_slot = (op, addr_m, src_m)
                else:
                    new_slot = slot
            elif engine == "flow":
                if op == "select":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, cond_m, a_m, b_m)
                elif op == "add_imm":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, a_m, imm)
                elif op == "jump_if":
                    new_slot = (op, cond_m, target)
                elif op == "vselect":
                    dest_m = map_r(dest)
                    count_change(dest, dest_m)
                    new_slot = (op, dest_m, cond_m, a_m, b_m)
                else:
                    new_slot = slot
            elif engine == "debug":
                if op == "compare":
                    new_slot = (op, loc_m, key)
                elif op == "vcompare":
                    new_slot = (op, loc_m, keys)
                else:
                    new_slot = slot
            else:
                new_slot = slot

            new_slots.append((engine, new_slot))

        slots = new_slots
        if debug_stats:
            print(
                f"renamer_changes: operands={renamed_operands} "
                f"unique_src_regs={len(renamed_regs)}"
            )
        if renamed_operands == 0:
            raise AssertionError(
                "rename_war_waw enabled but no scratch operands changed"
            )

    n = len(slots)
    deps: List[SlotDeps] = [
        _slot_deps(engine, slot, load_offset_reads_base=load_offset_reads_base)
        for engine, slot in slots
    ]

    # Build dependency graph.
    last_write: Dict[int, int] = {}
    last_access: Dict[int, int] = {}
    last_mem: int | None = None
    last_mem_by_key: Dict[tuple, int] = {}
    last_store_by_key: Dict[tuple, int] = {}
    base_key_by_reg: Dict[int, tuple[int, ...] | None] = {}

    # Scratch addresses that are never written inside this schedule are treated as read-only.
    all_writes = set()
    for dep in deps:
        all_writes.update(dep.writes)

    if disambiguate_mem:
        affine_key_by_reg: Dict[int, tuple[int, int, tuple[int, ...]] | None] = {}
        const_val_by_reg: Dict[int, int] = {}
        uint32_mod = 2**32

        def affine_key(reg: int) -> tuple[int, int, tuple[int, ...]] | None:
            key = affine_key_by_reg.get(reg)
            if key is not None:
                return key
            if reg not in all_writes:
                return (reg, 0, (reg,))
            return None

        def mem_key_for_addr(addr_reg: int) -> tuple:
            # Group by base stream (base + offset + readonly sources) when known.
            if addr_reg in const_val_by_reg:
                return ("const", const_val_by_reg[addr_reg])
            key = affine_key(addr_reg)
            if key is None:
                return ("reg", addr_reg)
            base_reg, imm_offset, readonly_sources = key
            return ("affine", base_reg, imm_offset, readonly_sources)

    else:

        def mem_key_for_addr(addr_reg: int) -> tuple:
            base_key = base_key_by_reg.get(addr_reg)
            if base_key is None:
                return ("reg", addr_reg)
            return ("base", addr_reg, base_key)

    succ: Dict[int, List[int]] = defaultdict(list)
    pred: List[List[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    raw_edges = 0
    war_edges = 0
    waw_edges = 0
    mem_edges = 0
    last_access_is_write: Dict[int, bool] = {}

    def add_edge(a: int, b: int) -> None:
        if a == b:
            return
        succ[a].append(b)
        pred[b].append(a)
        indeg[b] += 1

    for i, ((engine, slot), dep) in enumerate(zip(slots, deps)):
        # RAW: read after last write
        for r in dep.reads:
            if r in last_write:
                add_edge(last_write[r], i)
                raw_edges += 1
        # WAR/WAW: write after any prior access
        for w in dep.writes:
            if w in last_access:
                add_edge(last_access[w], i)
                if last_access_is_write.get(w, False):
                    waw_edges += 1
                else:
                    war_edges += 1
        # Memory ordering: either fully serialized or by address key.
        if dep.is_mem:
            if serialize_mem and last_mem is not None:
                add_edge(last_mem, i)
                mem_edges += 1
            elif dep.mem_addr_reg is not None:
                mem_key = mem_key_for_addr(dep.mem_addr_reg)
                if dep.is_store:
                    if mem_key in last_mem_by_key:
                        add_edge(last_mem_by_key[mem_key], i)
                        mem_edges += 1
                else:
                    if mem_key in last_store_by_key:
                        add_edge(last_store_by_key[mem_key], i)
                        mem_edges += 1
            elif last_mem is not None:
                # Unknown address; serialize with last mem op.
                add_edge(last_mem, i)
                mem_edges += 1

        # Update trackers
        for r in dep.reads:
            last_access[r] = i
            last_access_is_write[r] = False
        for w in dep.writes:
            last_access[w] = i
            last_access_is_write[w] = True
            last_write[w] = i
        if dep.is_mem:
            last_mem = i
            if dep.mem_addr_reg is not None:
                mem_key = mem_key_for_addr(dep.mem_addr_reg)
                last_mem_by_key[mem_key] = i
                if dep.is_store:
                    last_store_by_key[mem_key] = i

        # Track latest base key for address registers written in this slot.
        if dep.writes:
            if engine == "alu" and slot[0] == "+":
                _, dest, a1, a2 = slot
                readonly = []
                if a1 not in all_writes:
                    readonly.append(a1)
                if a2 not in all_writes:
                    readonly.append(a2)
                base_key_by_reg[dest] = tuple(sorted(readonly)) if readonly else None
            elif engine == "flow" and slot[0] == "add_imm":
                _, dest, a, _imm = slot
                base_key_by_reg[dest] = (a,) if a not in all_writes else None
            else:
                for w in dep.writes:
                    base_key_by_reg[w] = None

        # Optional affine tracking for improved memory disambiguation.
        if disambiguate_mem and dep.writes:
            const_updates: Dict[int, int] = {}
            affine_updates: Dict[int, tuple[int, int, tuple[int, ...]] | None] = {}

            if engine == "load" and slot[0] == "const":
                _, dest, val = slot
                const_updates[dest] = val % uint32_mod
            elif engine == "flow" and slot[0] == "add_imm":
                _, dest, a, imm = slot
                base_key = affine_key(a)
                if base_key is not None:
                    base_reg, base_off, sources = base_key
                    affine_updates[dest] = (
                        base_reg,
                        (base_off + imm) % uint32_mod,
                        sources,
                    )
                if a in const_val_by_reg:
                    const_updates[dest] = (const_val_by_reg[a] + imm) % uint32_mod
            elif engine == "alu" and slot[0] == "+":
                _, dest, a1, a2 = slot
                a1_const = const_val_by_reg.get(a1)
                a2_const = const_val_by_reg.get(a2)
                if a1_const is not None and a2_const is not None:
                    const_updates[dest] = (a1_const + a2_const) % uint32_mod

                a1_key = affine_key(a1)
                a2_key = affine_key(a2)
                new_key = None
                if a1_key is not None and a2_const is not None:
                    base_reg, base_off, sources = a1_key
                    new_key = (
                        base_reg,
                        (base_off + a2_const) % uint32_mod,
                        sources,
                    )
                elif a2_key is not None and a1_const is not None:
                    base_reg, base_off, sources = a2_key
                    new_key = (
                        base_reg,
                        (base_off + a1_const) % uint32_mod,
                        sources,
                    )
                elif a1_key is not None and a2 not in all_writes:
                    base_reg, base_off, sources = a1_key
                    merged = tuple(sorted(set(sources + (a2,))))
                    new_key = (base_reg, base_off, merged)
                elif a2_key is not None and a1 not in all_writes:
                    base_reg, base_off, sources = a2_key
                    merged = tuple(sorted(set(sources + (a1,))))
                    new_key = (base_reg, base_off, merged)
                if new_key is not None:
                    affine_updates[dest] = new_key
            elif engine == "alu" and slot[0] == "-":
                _, dest, a1, a2 = slot
                a1_const = const_val_by_reg.get(a1)
                a2_const = const_val_by_reg.get(a2)
                if a1_const is not None and a2_const is not None:
                    const_updates[dest] = (a1_const - a2_const) % uint32_mod
                a1_key = affine_key(a1)
                if a1_key is not None and a2_const is not None:
                    base_reg, base_off, sources = a1_key
                    affine_updates[dest] = (
                        base_reg,
                        (base_off - a2_const) % uint32_mod,
                        sources,
                    )

            for w in dep.writes:
                if w in const_updates:
                    const_val_by_reg[w] = const_updates[w]
                else:
                    const_val_by_reg.pop(w, None)
                if w in affine_updates:
                    affine_key_by_reg[w] = affine_updates[w]
                else:
                    affine_key_by_reg[w] = None

    if debug_stats:
        unique_preds = [len(set(p)) for p in pred]
        if unique_preds:
            avg_preds = sum(unique_preds) / len(unique_preds)
            print(
                "dep_graph: "
                f"raw={raw_edges} war={war_edges} waw={waw_edges} mem={mem_edges} "
                f"total={sum(indeg)}"
            )
            print(
                "unique_predecessors: "
                f"min={min(unique_preds)} avg={avg_preds:.2f} max={max(unique_preds)}"
            )

    if weighted_priority:
        if priority_weights is None:
            priority_weights = {
                "load": 3,
                "store": 2,
                "alu": 1,
                "valu": 3,
                "flow": 1,
                "debug": 1,
            }
        weights = [max(1, priority_weights.get(slots[i][0], 1)) for i in range(n)]
    else:
        weights = [1] * n

    # Priority score: (optionally weighted) longest path to exit (criticality).
    indeg_tmp = indeg.copy()
    topo: List[int] = []
    q = deque(i for i in range(n) if indeg_tmp[i] == 0)
    while q:
        node = q.popleft()
        topo.append(node)
        for nxt in succ.get(node, []):
            indeg_tmp[nxt] -= 1
            if indeg_tmp[nxt] == 0:
                q.append(nxt)
    topo_ok = len(topo) == n
    if len(topo) != n:
        # Graph should be acyclic; fall back to program order if not.
        topo = list(range(n))
    priority = [0] * n
    for node in reversed(topo):
        if succ.get(node):
            priority[node] = weights[node] + max(priority[nxt] for nxt in succ[node])
        else:
            priority[node] = weights[node]

    slack = [0] * n
    if slack_tie_break and topo_ok:
        # Earliest-start computation for slack (weighted by engine).
        asap = [0] * n
        for node in topo:
            if pred[node]:
                asap[node] = max(asap[p] + weights[p] for p in pred[node])
        makespan = 0
        for node in range(n):
            finish = asap[node] + priority[node]
            if finish > makespan:
                makespan = finish
        for node in range(n):
            slack[node] = makespan - (asap[node] + priority[node])

    # Ready queue (stable order)
    ready = deque(i for i in range(n) if indeg[i] == 0)
    scheduled = 0
    bundles: List[Dict[str, List[Tuple]]] = []

    engine_priority = ["load", "store", "alu", "valu", "flow", "debug"]

    def ready_sort_key(idx: int) -> Tuple[int, int]:
        # Higher priority first, then lower original index.
        if slack_tie_break:
            return (-priority[idx], slack[idx], idx)
        return (-priority[idx], idx)

    while scheduled < n:
        bundle: Dict[str, List[Tuple]] = {}
        used = defaultdict(int)
        picked: List[int] = []

        ready_full = list(ready)
        ready_list = ready_full
        if window and len(ready_list) > window:
            ready_list = ready_list[:window]
        picked_set = set()

        def pick_idx(idx: int) -> None:
            engine, slot = slots[idx]
            bundle.setdefault(engine, []).append(slot)
            used[engine] += 1
            picked.append(idx)
            picked_set.add(idx)

        if global_pick:
            candidates = [idx for idx in ready_list if idx not in picked_set]
            candidates.sort(key=ready_sort_key)
            for idx in candidates:
                engine = slots[idx][0]
                if used[engine] >= slot_limits.get(engine, 0):
                    continue
                pick_idx(idx)
        else:
            # Fill each engine from the ready set using priority score.
            for engine in engine_priority:
                limit = slot_limits.get(engine, 0)
                if limit == 0:
                    continue
                candidates = [
                    idx
                    for idx in ready_list
                    if idx not in picked_set and slots[idx][0] == engine
                ]
                if not candidates:
                    continue
                candidates.sort(key=ready_sort_key)
                for idx in candidates:
                    if used[engine] >= limit:
                        break
                    pick_idx(idx)

        # Opportunistic fill: use remaining capacity across engines by priority.
        if any(used[e] < slot_limits.get(e, 0) for e in engine_priority):
            remaining = [idx for idx in ready_full if idx not in picked_set]
            remaining.sort(key=ready_sort_key)
            for idx in remaining:
                engine = slots[idx][0]
                if used[engine] >= slot_limits.get(engine, 0):
                    continue
                pick_idx(idx)

        if bundle_repair and picked_set:
            # One-bundle lookahead: prefer ops that unlock high-priority successors.
            unlock_score: Dict[int, int] = {}
            for idx in ready_list:
                score = 0
                for s in succ.get(idx, []):
                    if indeg[s] == 1:
                        score += priority[s]
                unlock_score[idx] = score

            def repair_score(idx: int) -> int:
                return priority[idx] + repair_weight * unlock_score.get(idx, 0)

            ready_unpicked = [idx for idx in ready_list if idx not in picked_set]
            ready_unpicked.sort(
                key=lambda i: (-repair_score(i), -priority[i], i)
            )

            for engine in engine_priority:
                picked_engine = [i for i in picked_set if slots[i][0] == engine]
                if not picked_engine:
                    continue
                unpicked_engine = [
                    i for i in ready_unpicked if slots[i][0] == engine
                ]
                if not unpicked_engine:
                    continue
                picked_engine.sort(
                    key=lambda i: (repair_score(i), priority[i], i)
                )
                unpicked_engine.sort(
                    key=lambda i: (-repair_score(i), -priority[i], i)
                )
                swap_count = min(len(picked_engine), len(unpicked_engine))
                for k in range(swap_count):
                    out_idx = picked_engine[k]
                    in_idx = unpicked_engine[k]
                    if repair_score(in_idx) > repair_score(out_idx):
                        picked_set.remove(out_idx)
                        picked_set.add(in_idx)

            if picked_set:
                picked = sorted(picked_set)
                bundle = {}
                used = defaultdict(int)
                for engine in engine_priority:
                    limit = slot_limits.get(engine, 0)
                    if limit == 0:
                        continue
                    indices = [
                        idx for idx in picked if slots[idx][0] == engine
                    ]
                    if not indices:
                        continue
                    indices.sort()
                    for idx in indices:
                        if used[engine] >= limit:
                            break
                        slot_engine, slot = slots[idx]
                        bundle.setdefault(engine, []).append(slot)
                        used[engine] += 1

        if not picked:
            # If nothing was picked (e.g., only engines with 0 capacity), force-pick the first ready.
            pick_idx(ready_full[0])

        # Rebuild ready deque without picked items, preserving original order.
        ready = deque(i for i in ready if i not in picked_set)

        bundles.append(bundle)
        scheduled += len(picked)

        # Update indegrees
        for i in picked:
            for j in succ.get(i, []):
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)

    return bundles


__all__ = ["schedule_slots"]
