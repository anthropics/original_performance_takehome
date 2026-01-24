"""
Greedy list scheduler for VLIW instruction bundling.

This module is intentionally conservative: it preserves program order across
any shared scratch address, and serializes all memory ops to avoid aliasing
hazards. It is meant as a safe starting point for faster bundling, not the
final performance solution.
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


def _slot_deps(engine: str, slot: Tuple) -> SlotDeps:
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
            return SlotDeps(reads={loc}, writes=set(), is_mem=False)
        if slot[0] == "vcompare":
            _, loc, keys = slot
            return SlotDeps(
                reads=set(range(loc, loc + len(keys))), writes=set(), is_mem=False
            )
        return SlotDeps(set(), set(), False)

    op = slot[0]
    reads: set[int] = set()
    writes: set[int] = set()
    is_mem = False

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
        elif op == "vload":
            # ("vload", dest_vec, addr)
            reads.add(slot[2])
            writes.update(range(slot[1], slot[1] + VLEN))
            is_mem = True
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
        elif op == "vstore":
            # ("vstore", addr, src_vec)
            reads.add(slot[1])
            reads.update(range(slot[2], slot[2] + VLEN))
            is_mem = True
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

    return SlotDeps(reads=reads, writes=writes, is_mem=is_mem)


def schedule_slots(
    slots: Sequence[Slot],
    slot_limits: Dict[str, int],
    *,
    serialize_mem: bool = True,
) -> List[Dict[str, List[Tuple]]]:
    """
    Schedule a sequence of slots into VLIW bundles.

    This uses a greedy list scheduler with conservative dependency tracking.
    Returns a list of instruction bundles (dict of engine -> list of slots).
    """
    n = len(slots)
    deps: List[SlotDeps] = [_slot_deps(engine, slot) for engine, slot in slots]

    # Build dependency graph.
    last_write: Dict[int, int] = {}
    last_access: Dict[int, int] = {}
    last_mem: int | None = None

    succ: Dict[int, List[int]] = defaultdict(list)
    indeg = [0] * n

    def add_edge(a: int, b: int) -> None:
        if a == b:
            return
        succ[a].append(b)
        indeg[b] += 1

    for i, dep in enumerate(deps):
        # RAW: read after last write
        for r in dep.reads:
            if r in last_write:
                add_edge(last_write[r], i)
        # WAR/WAW: write after any prior access
        for w in dep.writes:
            if w in last_access:
                add_edge(last_access[w], i)
        # Memory ordering (very conservative)
        if serialize_mem and dep.is_mem and last_mem is not None:
            add_edge(last_mem, i)

        # Update trackers
        for r in dep.reads:
            last_access[r] = i
        for w in dep.writes:
            last_access[w] = i
            last_write[w] = i
        if dep.is_mem and serialize_mem:
            last_mem = i

    # Ready queue (stable order)
    ready = deque(i for i in range(n) if indeg[i] == 0)
    scheduled = 0
    bundles: List[Dict[str, List[Tuple]]] = []

    while scheduled < n:
        bundle: Dict[str, List[Tuple]] = {}
        used = defaultdict(int)
        picked: List[int] = []

        # Greedily pack from ready in order.
        for _ in range(len(ready)):
            i = ready[0]
            engine, slot = slots[i]
            limit = slot_limits.get(engine, 0)
            if used[engine] < limit:
                ready.popleft()
                bundle.setdefault(engine, []).append(slot)
                used[engine] += 1
                picked.append(i)
            else:
                # Rotate to end if can't fit this engine this cycle.
                ready.rotate(-1)

        if not picked:
            # No slot could fit due to engine limits; start a new cycle
            # by force-picking the first ready slot.
            i = ready.popleft()
            engine, slot = slots[i]
            bundle.setdefault(engine, []).append(slot)
            picked.append(i)

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
