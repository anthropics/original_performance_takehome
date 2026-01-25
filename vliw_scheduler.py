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

from problem import VLEN

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
) -> List[Dict[str, List[Tuple]]]:
    """
    Schedule a sequence of slots into VLIW bundles.

    This uses a greedy list scheduler with conservative dependency tracking.
    Priority is based on longest-path criticality in the dependency graph.
    Set disambiguate_mem=True to use affine address keys for memory ordering.
    Set load_offset_reads_base=True if load_offset uses base+imm addressing.
    Returns a list of instruction bundles (dict of engine -> list of slots).
    """
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
    indeg = [0] * n

    def add_edge(a: int, b: int) -> None:
        if a == b:
            return
        succ[a].append(b)
        indeg[b] += 1

    for i, ((engine, slot), dep) in enumerate(zip(slots, deps)):
        # RAW: read after last write
        for r in dep.reads:
            if r in last_write:
                add_edge(last_write[r], i)
        # WAR/WAW: write after any prior access
        for w in dep.writes:
            if w in last_access:
                add_edge(last_access[w], i)
        # Memory ordering: either fully serialized or by address key.
        if dep.is_mem:
            if serialize_mem and last_mem is not None:
                add_edge(last_mem, i)
            elif dep.mem_addr_reg is not None:
                mem_key = mem_key_for_addr(dep.mem_addr_reg)
                if dep.is_store:
                    if mem_key in last_mem_by_key:
                        add_edge(last_mem_by_key[mem_key], i)
                else:
                    if mem_key in last_store_by_key:
                        add_edge(last_store_by_key[mem_key], i)
            elif last_mem is not None:
                # Unknown address; serialize with last mem op.
                add_edge(last_mem, i)

        # Update trackers
        for r in dep.reads:
            last_access[r] = i
        for w in dep.writes:
            last_access[w] = i
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

            for w in dep.writes:
                if w in const_updates:
                    const_val_by_reg[w] = const_updates[w]
                else:
                    const_val_by_reg.pop(w, None)
                if w in affine_updates:
                    affine_key_by_reg[w] = affine_updates[w]
                else:
                    affine_key_by_reg[w] = None

    # Priority score: longest path to exit (criticality).
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
    if len(topo) != n:
        # Graph should be acyclic; fall back to program order if not.
        topo = list(range(n))
    priority = [0] * n
    for node in reversed(topo):
        if succ.get(node):
            priority[node] = 1 + max(priority[nxt] for nxt in succ[node])

    # Ready queue (stable order)
    ready = deque(i for i in range(n) if indeg[i] == 0)
    scheduled = 0
    bundles: List[Dict[str, List[Tuple]]] = []

    engine_priority = ["load", "store", "alu", "valu", "flow", "debug"]

    def ready_sort_key(idx: int) -> Tuple[int, int]:
        # Higher priority first, then lower original index.
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
