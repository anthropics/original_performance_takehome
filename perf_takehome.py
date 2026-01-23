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
from dataclasses import dataclass, field
import random
import unittest
from typing import Optional

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


# ============================================================================
# Dependency Graph and Instruction Scheduler
# ============================================================================

@dataclass
class InstrNode:
    """Represents a single instruction in the dependency graph."""
    id: int
    engine: str
    slot: tuple
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    batch_item: int = 0
    round_num: int = 0
    in_degree: int = 0
    successors: list = field(default_factory=list)


class DependencyGraph:
    """DAG of instruction dependencies with Kahn's algorithm."""

    def __init__(self):
        self.nodes: dict[int, InstrNode] = {}

    def add_node(self, node: InstrNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, from_id: int, to_id: int) -> None:
        if to_id not in self.nodes[from_id].successors:
            self.nodes[from_id].successors.append(to_id)
            self.nodes[to_id].in_degree += 1

    def topo_sort_kahn(self) -> list[list[int]]:
        in_degree = {nid: node.in_degree for nid, node in self.nodes.items()}
        ready = [nid for nid, deg in in_degree.items() if deg == 0]
        levels = []

        while ready:
            levels.append(ready)
            next_ready = []
            for nid in ready:
                for succ in self.nodes[nid].successors:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        next_ready.append(succ)
            ready = next_ready

        return levels


def extract_reads_writes(engine: str, slot: tuple) -> tuple[set, set]:
    """Analyze instruction to determine scratch addresses accessed."""
    reads, writes = set(), set()

    if engine == "alu":
        if slot[0] == "fma":
            _, dest, a, b, c = slot
            writes.add(dest)
            reads.update({a, b, c})
        else:
            _, dest, a1, a2 = slot
            writes.add(dest)
            reads.update({a1, a2})

    elif engine == "valu":
        if slot[0] == "vbroadcast":
            _, dest, src = slot
            reads.add(src)
            for i in range(VLEN):
                writes.add(dest + i)
        elif slot[0] == "multiply_add":
            _, dest, a, b, c = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.update({a + i, b + i, c + i})
        else:
            _, dest, a1, a2 = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.update({a1 + i, a2 + i})

    elif engine == "load":
        if slot[0] == "load":
            _, dest, addr = slot
            writes.add(dest)
            reads.add(addr)
        elif slot[0] == "vload":
            _, dest, addr = slot
            reads.add(addr)
            for i in range(VLEN):
                writes.add(dest + i)
        elif slot[0] == "const":
            _, dest, _ = slot
            writes.add(dest)

    elif engine == "store":
        if slot[0] == "store":
            _, addr, src = slot
            reads.update({addr, src})
        elif slot[0] == "vstore":
            _, addr, src = slot
            reads.add(addr)
            for i in range(VLEN):
                reads.add(src + i)

    elif engine == "flow":
        if slot[0] == "select":
            _, dest, cond, a, b = slot
            writes.add(dest)
            reads.update({cond, a, b})
        elif slot[0] == "vselect":
            _, dest, cond, a, b = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.update({cond + i, a + i, b + i})

    elif engine == "debug":
        if slot[0] == "compare":
            _, addr, _ = slot
            reads.add(addr)

    return reads, writes


def build_dependency_graph(instructions: list[tuple]) -> DependencyGraph:
    """Build dependency graph from tagged instructions."""
    graph = DependencyGraph()
    last_writer: dict[int, int] = {}
    last_readers: dict[int, list[int]] = defaultdict(list)

    for i, (engine, slot, batch_item, round_num) in enumerate(instructions):
        reads, writes = extract_reads_writes(engine, slot)
        node = InstrNode(id=i, engine=engine, slot=slot, reads=reads, writes=writes,
                        batch_item=batch_item, round_num=round_num)
        graph.add_node(node)

        # RAW
        for addr in reads:
            if addr in last_writer:
                graph.add_edge(last_writer[addr], i)

        # WAW
        for addr in writes:
            if addr in last_writer:
                graph.add_edge(last_writer[addr], i)

        # WAR
        for addr in writes:
            for reader_id in last_readers[addr]:
                if reader_id != last_writer.get(addr):
                    graph.add_edge(reader_id, i)

        for addr in writes:
            last_writer[addr] = i
            last_readers[addr] = []
        for addr in reads:
            if i not in last_readers[addr]:
                last_readers[addr].append(i)

    return graph


def add_stagger_dependencies(graph: DependencyGraph, wave_size: int = 8):
    """Add stagger dependencies for software pipelining."""
    by_item_round = defaultdict(list)
    for nid, node in graph.nodes.items():
        if node.engine != "debug":
            by_item_round[(node.batch_item, node.round_num)].append(node)

    for key in by_item_round:
        by_item_round[key].sort(key=lambda n: n.id)

    first_node = {}
    for key, nodes in by_item_round.items():
        if nodes:
            first_node[key] = nodes[0]

    items = sorted(set(k[0] for k in by_item_round.keys()))
    rounds = sorted(set(k[1] for k in by_item_round.keys()))

    for round_num in rounds:
        for item in items:
            wave = item // wave_size
            if wave > 0:
                prev_item = (wave - 1) * wave_size
                key_curr = (item, round_num)
                key_prev = (prev_item, round_num)
                if key_curr in first_node and key_prev in first_node:
                    graph.add_edge(first_node[key_prev].id, first_node[key_curr].id)


def pack_levels_into_bundles(graph: DependencyGraph, levels: list[list[int]],
                             broadcast_state: dict) -> list[dict]:
    """Pack instructions into VLIW bundles with vectorization."""
    bundles = []
    broadcast_cache = broadcast_state['cache']
    broadcast_ptr = broadcast_state['ptr']

    add_stagger_dependencies(graph, wave_size=8)

    completed = set()
    remaining_in_degree = {nid: node.in_degree for nid, node in graph.nodes.items()}
    ready = [nid for nid, deg in remaining_in_degree.items() if deg == 0]

    while ready:
        by_engine = defaultdict(list)
        for nid in ready:
            node = graph.nodes[nid]
            by_engine[node.engine].append(node)

        # Vectorize ALU ops
        alu_nodes = by_engine.get("alu", [])
        valu_slots = []
        remaining_alu = []
        pre_bundle_valu = []

        by_op = defaultdict(list)
        for node in alu_nodes:
            by_op[node.slot[0]].append(node)

        for op, nodes in by_op.items():
            nodes_sorted = sorted(nodes, key=lambda n: n.slot[1])
            i = 0
            while i < len(nodes_sorted):
                if i + VLEN <= len(nodes_sorted):
                    group = nodes_sorted[i:i+VLEN]
                    dests = sorted(n.slot[1] for n in group)
                    if dests == list(range(dests[0], dests[0] + VLEN)):
                        # Can vectorize
                        if op == "fma":
                            # FMA vectorization
                            nodes_by_dest = sorted(group, key=lambda n: n.slot[1])
                            dest_base = nodes_by_dest[0].slot[1]
                            src_a_base = nodes_by_dest[0].slot[2]
                            src_b = nodes_by_dest[0].slot[3]
                            src_c = nodes_by_dest[0].slot[4]

                            # Handle broadcasts if needed
                            if all(n.slot[3] == src_b for n in group):
                                if src_b not in broadcast_cache:
                                    vec_addr = broadcast_ptr
                                    broadcast_ptr += VLEN
                                    broadcast_cache[src_b] = vec_addr
                                    pre_bundle_valu.append(("vbroadcast", vec_addr, src_b))
                                src_b = broadcast_cache[src_b]

                            if all(n.slot[4] == src_c for n in group):
                                if src_c not in broadcast_cache:
                                    vec_addr = broadcast_ptr
                                    broadcast_ptr += VLEN
                                    broadcast_cache[src_c] = vec_addr
                                    pre_bundle_valu.append(("vbroadcast", vec_addr, src_c))
                                src_c = broadcast_cache[src_c]

                            valu_slots.append((("multiply_add", dest_base, src_a_base, src_b, src_c), [n.id for n in group]))
                            i += VLEN
                            continue
                        else:
                            # Standard ALU vectorization
                            nodes_by_dest = sorted(group, key=lambda n: n.slot[1])
                            dest_base = nodes_by_dest[0].slot[1]
                            src1_base = nodes_by_dest[0].slot[2]
                            src2_base = nodes_by_dest[0].slot[3]

                            # Check for broadcasts
                            if all(n.slot[2] == src1_base for n in group):
                                if src1_base not in broadcast_cache:
                                    vec_addr = broadcast_ptr
                                    broadcast_ptr += VLEN
                                    broadcast_cache[src1_base] = vec_addr
                                    pre_bundle_valu.append(("vbroadcast", vec_addr, src1_base))
                                src1_base = broadcast_cache[src1_base]

                            if all(n.slot[3] == src2_base for n in group):
                                if src2_base not in broadcast_cache:
                                    vec_addr = broadcast_ptr
                                    broadcast_ptr += VLEN
                                    broadcast_cache[src2_base] = vec_addr
                                    pre_bundle_valu.append(("vbroadcast", vec_addr, src2_base))
                                src2_base = broadcast_cache[src2_base]

                            valu_slots.append(((op, dest_base, src1_base, src2_base), [n.id for n in group]))
                            i += VLEN
                            continue

                remaining_alu.append(nodes_sorted[i])
                i += 1

        by_engine["alu"] = remaining_alu

        # Emit broadcasts
        for bi in range(0, len(pre_bundle_valu), SLOT_LIMITS["valu"]):
            batch = pre_bundle_valu[bi:bi+SLOT_LIMITS["valu"]]
            bundles.append({"valu": batch})

        for valu_slot, node_ids in valu_slots:
            by_engine["valu"].append((valu_slot, node_ids))

        # Pack bundles
        processed_this_round = set()
        while any(by_engine.values()):
            bundle = {}
            bundle_nodes = []

            for engine in ["alu", "valu", "load", "store", "flow", "debug"]:
                if by_engine.get(engine):
                    limit = SLOT_LIMITS.get(engine, 1)
                    take = min(limit, len(by_engine[engine]))
                    taken = []
                    for _ in range(take):
                        item = by_engine[engine].pop(0)
                        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], list):
                            taken.append(item[0])
                            bundle_nodes.extend(item[1])
                        else:
                            taken.append(item.slot)
                            bundle_nodes.append(item.id)
                    if taken:
                        bundle[engine] = taken

            if bundle:
                bundles.append(bundle)
                processed_this_round.update(bundle_nodes)

        completed.update(processed_this_round)

        ready = []
        for nid in processed_this_round:
            for succ in graph.nodes[nid].successors:
                if succ not in completed:
                    remaining_in_degree[succ] -= 1
                    if remaining_in_degree[succ] == 0:
                        ready.append(succ)

    broadcast_state['ptr'] = broadcast_ptr
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

    def build(self, slots: list[tuple[Engine, tuple]]):
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

    def build_hash_tagged(self, val_addr, tmp1, tmp2, round_num, batch_item):
        """Build hash with FMA optimization."""
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # FMA: val = val * (1 + 2^shift) + const
                multiplier = 1 + (1 << val3)
                mult_const = self.scratch_const(multiplier)
                const1 = self.scratch_const(val1)
                slots.append(("alu", ("fma", val_addr, val_addr, mult_const, const1), batch_item, round_num))
            else:
                slots.append(("alu", (op1, tmp1, val_addr, self.scratch_const(val1)), batch_item, round_num))
                slots.append(("alu", (op3, tmp2, val_addr, self.scratch_const(val3)), batch_item, round_num))
                slots.append(("alu", (op2, val_addr, tmp1, tmp2), batch_item, round_num))
            slots.append(("debug", ("compare", val_addr, (round_num, batch_item, "hash_stage", hi)), batch_item, round_num))
        return slots

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """Optimized kernel with vectorization and software pipelining."""
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                    "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        tagged_body = []
        GROUP_SIZE = 128

        var_names = ["tmp_idx", "tmp_val", "node_val", "tmp_addr", "hash_tmp"]
        var_bases = {}
        for var in var_names:
            var_bases[var] = self.alloc_scratch(f"{var}_vec", GROUP_SIZE)

        batch_scratch = {i: {var: var_bases[var] + i for var in var_names} for i in range(GROUP_SIZE)}

        broadcast_base = self.alloc_scratch("broadcast", 20 * VLEN)
        broadcast_state = {'cache': {}, 'ptr': broadcast_base}

        vload_bases = [self.alloc_scratch(f"vl_{j}") for j in range(GROUP_SIZE // VLEN * 2)]

        # Shared tree values
        shared_node_val = tmp3
        tree1_addr = self.alloc_scratch("tree1")
        tree2_addr = self.alloc_scratch("tree2")
        tree1_vec = self.alloc_scratch("tree1_v", VLEN)
        tree2_vec = self.alloc_scratch("tree2_v", VLEN)
        one_vec = self.alloc_scratch("one_v", VLEN)

        for group_start in range(0, batch_size, GROUP_SIZE):
            group_end = min(group_start + GROUP_SIZE, batch_size)
            group_size = group_end - group_start

            # Load initial values
            for ci in range(0, group_size, VLEN):
                cs = group_start + ci
                cc = self.scratch_const(cs)
                vbi = vload_bases[(ci // VLEN) * 2]
                vbv = vload_bases[(ci // VLEN) * 2 + 1]
                tagged_body.append({"alu": [("+", vbi, self.scratch["inp_indices_p"], cc),
                                           ("+", vbv, self.scratch["inp_values_p"], cc)]})
                tagged_body.append({"load": [("vload", var_bases["tmp_idx"] + ci, vbi),
                                            ("vload", var_bases["tmp_val"] + ci, vbv)]})

            # Load shared values
            tagged_body.append({"load": [("load", shared_node_val, self.scratch["forest_values_p"])]})
            tagged_body.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], one_const),
                                       ("+", tmp2, self.scratch["forest_values_p"], two_const)]})
            tagged_body.append({"load": [("load", tree1_addr, tmp1), ("load", tree2_addr, tmp2)]})
            tagged_body.append({"valu": [("vbroadcast", tree1_vec, tree1_addr),
                                        ("vbroadcast", tree2_vec, tree2_addr),
                                        ("vbroadcast", one_vec, one_const)]})

            group_body = []
            for round_num in range(rounds):
                if round_num == 0:
                    # All items at index 0
                    for i in range(group_size):
                        item = group_start + i
                        bs = batch_scratch[i]
                        group_body.append(("alu", ("^", bs["tmp_val"], bs["tmp_val"], shared_node_val), item, round_num))
                        group_body.extend(self.build_hash_tagged(bs["tmp_val"], bs["hash_tmp"], bs["node_val"], round_num, item))
                        group_body.append(("alu", ("&", bs["tmp_addr"], bs["tmp_val"], one_const), item, round_num))
                        group_body.append(("alu", ("*", bs["tmp_idx"], bs["tmp_idx"], two_const), item, round_num))
                        group_body.append(("alu", ("+", bs["tmp_idx"], bs["tmp_idx"], one_const), item, round_num))
                        group_body.append(("alu", ("+", bs["tmp_idx"], bs["tmp_idx"], bs["tmp_addr"]), item, round_num))
                        group_body.append(("alu", ("<", bs["hash_tmp"], bs["tmp_idx"], self.scratch["n_nodes"]), item, round_num))
                        group_body.append(("alu", ("*", bs["tmp_idx"], bs["tmp_idx"], bs["hash_tmp"]), item, round_num))

                elif round_num == 1:
                    # Items at index 1 or 2 - use vselect
                    for ci in range(0, group_size, VLEN):
                        item = group_start + ci
                        idx_v = var_bases["tmp_idx"] + ci
                        val_v = var_bases["tmp_val"] + ci
                        nv_v = var_bases["node_val"] + ci
                        cond_v = var_bases["tmp_addr"] + ci
                        group_body.append(("valu", ("-", cond_v, idx_v, one_vec), item, round_num))
                        group_body.append(("flow", ("vselect", nv_v, cond_v, tree2_vec, tree1_vec), item, round_num))
                        group_body.append(("valu", ("^", val_v, val_v, nv_v), item, round_num))

                    for i in range(group_size):
                        item = group_start + i
                        bs = batch_scratch[i]
                        group_body.extend(self.build_hash_tagged(bs["tmp_val"], bs["hash_tmp"], bs["node_val"], round_num, item))
                        group_body.append(("alu", ("&", bs["tmp_addr"], bs["tmp_val"], one_const), item, round_num))
                        group_body.append(("alu", ("*", bs["tmp_idx"], bs["tmp_idx"], two_const), item, round_num))
                        group_body.append(("alu", ("+", bs["tmp_idx"], bs["tmp_idx"], one_const), item, round_num))
                        group_body.append(("alu", ("+", bs["tmp_idx"], bs["tmp_idx"], bs["tmp_addr"]), item, round_num))
                        group_body.append(("alu", ("<", bs["hash_tmp"], bs["tmp_idx"], self.scratch["n_nodes"]), item, round_num))
                        group_body.append(("alu", ("*", bs["tmp_idx"], bs["tmp_idx"], bs["hash_tmp"]), item, round_num))

                else:
                    # Scattered loads
                    for i in range(group_size):
                        item = group_start + i
                        bs = batch_scratch[i]
                        group_body.append(("alu", ("+", bs["tmp_addr"], self.scratch["forest_values_p"], bs["tmp_idx"]), item, round_num))
                        group_body.append(("load", ("load", bs["node_val"], bs["tmp_addr"]), item, round_num))
                        group_body.append(("alu", ("^", bs["tmp_val"], bs["tmp_val"], bs["node_val"]), item, round_num))
                        group_body.extend(self.build_hash_tagged(bs["tmp_val"], bs["hash_tmp"], bs["node_val"], round_num, item))
                        group_body.append(("alu", ("&", bs["tmp_addr"], bs["tmp_val"], one_const), item, round_num))
                        group_body.append(("alu", ("*", bs["tmp_idx"], bs["tmp_idx"], two_const), item, round_num))
                        group_body.append(("alu", ("+", bs["tmp_idx"], bs["tmp_idx"], one_const), item, round_num))
                        group_body.append(("alu", ("+", bs["tmp_idx"], bs["tmp_idx"], bs["tmp_addr"]), item, round_num))
                        group_body.append(("alu", ("<", bs["hash_tmp"], bs["tmp_idx"], self.scratch["n_nodes"]), item, round_num))
                        group_body.append(("alu", ("*", bs["tmp_idx"], bs["tmp_idx"], bs["hash_tmp"]), item, round_num))

            graph = build_dependency_graph(group_body)
            levels = graph.topo_sort_kahn()
            instrs = pack_levels_into_bundles(graph, levels, broadcast_state)
            tagged_body.extend(instrs)

            # Store results
            for ci in range(0, group_size, VLEN):
                cs = group_start + ci
                cc = self.scratch_const(cs)
                vbi = vload_bases[(ci // VLEN) * 2]
                vbv = vload_bases[(ci // VLEN) * 2 + 1]
                tagged_body.append({"alu": [("+", vbi, self.scratch["inp_indices_p"], cc),
                                           ("+", vbv, self.scratch["inp_values_p"], cc)]})
                tagged_body.append({"store": [("vstore", vbi, var_bases["tmp_idx"] + ci),
                                             ("vstore", vbv, var_bases["tmp_val"] + ci)]})

        self.instrs.extend(tagged_body)
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(forest_height: int, rounds: int, batch_size: int,
                  seed: int = 123, trace: bool = False, prints: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES,
                     value_trace=value_trace, trace=trace)
    machine.prints = prints

    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        assert (machine.mem[inp_values_p:inp_values_p + len(inp.values)] ==
                ref_mem[inp_values_p:inp_values_p + len(inp.values)]), f"Incorrect on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5]:mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6]:mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
