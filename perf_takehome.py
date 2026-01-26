import heapq
import random
from collections import defaultdict

from problem import (
    HASH_STAGES,
    N_CORES,
    SCRATCH_SIZE,
    SLOT_LIMITS,
    VLEN,
    DebugInfo,
    Input,
    Machine,
    Tree,
    build_mem_image,
    reference_kernel2,
)


def _vec_range(base, length=VLEN):
    return range(base, base + length)


def _slot_rw(engine, slot):
    reads, writes = [], []
    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads, writes = [a1, a2], [dest]
    elif engine == "valu":
        match slot:
            case ("vbroadcast", dest, src):
                reads, writes = [src], list(_vec_range(dest))
            case ("multiply_add", dest, a, b, c):
                reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
                writes = list(_vec_range(dest))
            case (_op, dest, a1, a2):
                reads = list(_vec_range(a1)) + list(_vec_range(a2))
                writes = list(_vec_range(dest))
    elif engine == "load":
        match slot:
            case ("load", dest, addr): reads, writes = [addr], [dest]
            case ("vload", dest, addr): reads, writes = [addr], list(_vec_range(dest))
            case ("const", dest, _val): writes = [dest]
    elif engine == "store":
        match slot:
            case ("store", addr, src): reads = [addr, src]
            case ("vstore", addr, src): reads = [addr] + list(_vec_range(src))
    elif engine == "flow":
        match slot:
            case ("select", dest, cond, a, b): reads, writes = [cond, a, b], [dest]
            case ("vselect", dest, cond, a, b):
                reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
                writes = list(_vec_range(dest))
            case _: pass
    return reads, writes


def _schedule_slots(slots):
    n_slots = len(slots)
    reads_list = [None] * n_slots
    writes_list = [None] * n_slots
    for i, (engine, slot) in enumerate(slots):
        reads, writes = _slot_rw(engine, slot)
        reads_list[i] = set(reads)
        writes_list[i] = set(writes)

    succs = [dict() for _ in range(n_slots)]
    indeg = [0] * n_slots
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(lambda: -1)

    def add_edge(src, dst, weight):
        existing = succs[src].get(dst)
        if existing is None:
            succs[src][dst] = weight
            indeg[dst] += 1
        elif weight > existing:
            succs[src][dst] = weight

    for i in range(n_slots):
        reads = reads_list[i]
        writes = writes_list[i]
        for a in reads:
            j = last_write[a]
            if j != -1:
                add_edge(j, i, 1)
        for a in writes:
            j = last_write[a]
            if j != -1:
                add_edge(j, i, 0)
            j = last_read[a]
            if j != -1:
                add_edge(j, i, 0)
        for a in reads:
            last_read[a] = i
        for a in writes:
            last_write[a] = i

    height = [1] * n_slots
    for i in range(n_slots - 1, -1, -1):
        if succs[i]:
            height[i] = max(1, max(1 + height[j] for j in succs[i]))

    ready_time = [0] * n_slots
    ready_set = {i for i in range(n_slots) if indeg[i] == 0}
    scheduled = [False] * n_slots

    engine_order = ("load", "flow", "valu", "alu", "store", "debug")
    cycles = []
    cycle = 0
    scheduled_count = 0

    alpha = 23
    while scheduled_count < n_slots:
        if not ready_set:
            raise RuntimeError("No schedulable slots; dependency cycle detected")
        engine_usage = defaultdict(int)
        heaps = {eng: [] for eng in SLOT_LIMITS}
        for i in ready_set:
            if ready_time[i] <= cycle:
                eng = slots[i][0]
                score = i - alpha * height[i]
                heapq.heappush(heaps[eng], (score, i))

        scheduled_any = False
        while True:
            progressed = False
            for eng in engine_order:
                if engine_usage[eng] >= SLOT_LIMITS[eng]:
                    continue
                heap = heaps.get(eng)
                if not heap:
                    continue
                while heap and scheduled[heap[0][1]]:
                    heapq.heappop(heap)
                if not heap:
                    continue
                _prio, idx = heapq.heappop(heap)
                scheduled[idx] = True
                scheduled_count += 1
                ready_set.remove(idx)
                while len(cycles) <= cycle:
                    cycles.append({})
                cycles[cycle].setdefault(eng, []).append(slots[idx][1])
                engine_usage[eng] += 1
                scheduled_any = True
                progressed = True

                for succ, weight in succs[idx].items():
                    indeg[succ] -= 1
                    ready_time[succ] = max(ready_time[succ], cycle + weight)
                    if indeg[succ] == 0:
                        ready_set.add(succ)
                        if ready_time[succ] <= cycle:
                            score = succ - alpha * height[succ]
                            heapq.heappush(
                                heaps[slots[succ][0]],
                                (score, succ),
                            )
            if not progressed:
                break

        if not scheduled_any:
            cycle = min(ready_time[i] for i in ready_set)
            continue
        cycle += 1

    return [c for c in cycles if c]


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None, slots=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            if slots is None:
                self.instrs.append({"load": [("const", addr, val)]})
            else:
                slots.append(("load", ("const", addr, val)))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None, slots=None):
        if val not in self.vconst_map:
            scalar = self.scratch_const(val, slots=slots)
            addr = self.alloc_vec(name)
            if slots is None:
                self.instrs.append({"valu": [("vbroadcast", addr, scalar)]})
            else:
                slots.append(("valu", ("vbroadcast", addr, scalar)))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds,
                     group_size=17, round_tile=14, mini_batch=1):
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")
        self.alloc_scratch("tmp_init")
        self.alloc_scratch("tmp_init2")

        FOREST_VALUES_P = 7
        INP_INDICES_P = 7 + n_nodes
        INP_VALUES_P = 7 + n_nodes + batch_size

        for v in ["forest_values_p", "inp_indices_p", "inp_values_p"]:
            self.alloc_scratch(v, 1)

        init_slots = []
        init_slots.append(("load", ("const", self.scratch["forest_values_p"], FOREST_VALUES_P)))
        init_slots.append(("load", ("const", self.scratch["inp_indices_p"], INP_INDICES_P)))
        init_slots.append(("load", ("const", self.scratch["inp_values_p"], INP_VALUES_P)))

        one_vec = self.scratch_vconst(1, "v_one", init_slots)
        two_vec = self.scratch_vconst(2, "v_two", init_slots)
        one_const = self.scratch_const(1, slots=init_slots)
        two_const = self.scratch_const(2, slots=init_slots)

        forest_vec = self.alloc_vec("v_forest_p")
        init_slots.append(("valu", ("vbroadcast", forest_vec, self.scratch["forest_values_p"])))

        three_vec = self.scratch_vconst(3, "v_three", init_slots)
        four_vec = self.scratch_vconst(4, "v_four", init_slots)
        seven_vec = self.scratch_vconst(7, "v_seven", init_slots)

        num_preload_nodes = min(15, n_nodes)
        node_vecs = []
        node_vecs_xor = []
        for node_idx in range(num_preload_nodes):
            node_scalar = self.alloc_scratch(f"node_{node_idx}")
            node_vec = self.alloc_vec(f"v_node_{node_idx}")
            node_vec_xor = self.alloc_vec(f"v_node_x_{node_idx}")
            node_offset = self.scratch_const(node_idx, slots=init_slots)
            addr_reg = tmp_addr if node_idx % 2 == 0 else tmp_addr2
            init_slots.append(("alu", ("+", addr_reg, self.scratch["forest_values_p"], node_offset)))
            init_slots.append(("load", ("load", node_scalar, addr_reg)))
            init_slots.append(("valu", ("vbroadcast", node_vec, node_scalar)))
            node_vecs.append(node_vec)
            node_vecs_xor.append(node_vec_xor)

        hash_vec_consts1, hash_vec_consts3, hash_mul_vecs = [], [], []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_vec_consts1.append(self.scratch_vconst(val1, slots=init_slots))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                hash_vec_consts3.append(None)
                hash_mul_vecs.append(self.scratch_vconst(1 + (1 << val3), slots=init_slots))
            else:
                hash_vec_consts3.append(self.scratch_vconst(val3, slots=init_slots))
                hash_mul_vecs.append(None)

        c12_val = (0x165667B1 + 0xD3A2646C) % (2**32)
        c1_512_val = (0x165667B1 << 9) % (2**32)
        vec_c12 = self.scratch_vconst(c12_val, "v_c12", init_slots)
        vec_c1_512 = self.scratch_vconst(c1_512_val, "v_c1_512", init_slots)
        vec_mul16896 = self.scratch_vconst(16896, "v_mul16896", init_slots)
        vec_33 = hash_mul_vecs[2]
        vec_stage5_const = hash_vec_consts1[-1]
        for node_vec, node_vec_xor in zip(node_vecs, node_vecs_xor):
            init_slots.append(("valu", ("^", node_vec_xor, node_vec, vec_stage5_const)))

        blocks_per_round = batch_size // VLEN
        idx_base = self.alloc_scratch("idx_scratch", batch_size)
        val_base = self.alloc_scratch("val_scratch", batch_size)

        offset = self.alloc_scratch("offset")
        init_slots.append(("load", ("const", offset, 0)))
        vlen_const = self.scratch_const(VLEN, slots=init_slots)

        slots = list(init_slots)

        for block in range(blocks_per_round):
            slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], offset)))
            slots.append(("load", ("vload", idx_base + block * VLEN, tmp_addr)))
            slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], offset)))
            slots.append(("load", ("vload", val_base + block * VLEN, tmp_addr)))
            slots.append(("alu", ("+", offset, offset, vlen_const)))

        contexts = []
        for _ in range(group_size):
            contexts.append({
                "node": self.alloc_vec(),
                "tmp1": self.alloc_vec(),
                "tmp2": self.alloc_vec(),
                "tmp3": self.alloc_vec(),
            })

        def emit_tree_lookup(ctx, idx_vec, val_vec, level, use_xor_nodes):
            nodes = node_vecs_xor if use_xor_nodes else node_vecs
            if level == 0:
                for lane in range(VLEN):
                    slots.append(("alu", ("^", val_vec + lane, val_vec + lane, nodes[0] + lane)))
            elif level == 1:
                slots.append(("valu", ("&", ctx["tmp1"], idx_vec, one_vec)))
                slots.append(("flow", ("vselect", ctx["node"], ctx["tmp1"], nodes[1], nodes[2])))
                for lane in range(VLEN):
                    slots.append(("alu", ("^", val_vec + lane, val_vec + lane, ctx["node"] + lane)))
            elif level == 2:
                slots.append(("valu", ("&", ctx["tmp1"], idx_vec, one_vec)))
                slots.append(("valu", ("&", ctx["tmp2"], idx_vec, two_vec)))
                slots.append(("flow", ("vselect", ctx["tmp3"], ctx["tmp2"], nodes[6], nodes[4])))
                slots.append(("flow", ("vselect", ctx["node"], ctx["tmp2"], nodes[3], nodes[5])))
                slots.append(("flow", ("vselect", ctx["node"], ctx["tmp1"], ctx["node"], ctx["tmp3"])))
                slots.append(("valu", ("^", val_vec, val_vec, ctx["node"])))
            elif level == 3:
                slots.append(("valu", ("-", ctx["tmp1"], idx_vec, seven_vec)))
                slots.append(("valu", ("&", ctx["tmp2"], ctx["tmp1"], one_vec)))
                slots.append(("valu", ("&", ctx["tmp3"], ctx["tmp1"], two_vec)))
                slots.append(("flow", ("vselect", ctx["node"], ctx["tmp2"], nodes[8], nodes[7])))
                slots.append(("flow", ("vselect", ctx["tmp1"], ctx["tmp2"], nodes[10], nodes[9])))
                slots.append(("flow", ("vselect", ctx["tmp1"], ctx["tmp3"], ctx["tmp1"], ctx["node"])))
                slots.append(("flow", ("vselect", ctx["node"], ctx["tmp2"], nodes[12], nodes[11])))
                slots.append(("flow", ("vselect", ctx["tmp2"], ctx["tmp2"], nodes[14], nodes[13])))
                slots.append(("flow", ("vselect", ctx["node"], ctx["tmp3"], ctx["tmp2"], ctx["node"])))
                slots.append(("valu", ("-", ctx["tmp3"], idx_vec, seven_vec)))
                slots.append(("valu", ("&", ctx["tmp3"], ctx["tmp3"], four_vec)))
                slots.append(("flow", ("vselect", ctx["node"], ctx["tmp3"], ctx["node"], ctx["tmp1"])))
                slots.append(("valu", ("^", val_vec, val_vec, ctx["node"])))
            else:
                for lane in range(VLEN):
                    slots.append(("alu", ("+", ctx["tmp1"] + lane, forest_vec + lane, idx_vec + lane)))
                for lane in range(VLEN):
                    slots.append(("load", ("load", ctx["node"] + lane, ctx["tmp1"] + lane)))
                for lane in range(VLEN):
                    slots.append(("alu", ("^", val_vec + lane, val_vec + lane, ctx["node"] + lane)))

        def emit_hash(ctx, val_vec, defer_const):
            for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                if hi == 2:
                    slots.append(("valu", ("multiply_add", ctx["tmp1"], val_vec, vec_mul16896, vec_c1_512)))
                    slots.append(("valu", ("multiply_add", val_vec, val_vec, vec_33, vec_c12)))
                    slots.append(("valu", ("^", val_vec, val_vec, ctx["tmp1"])))
                elif hi == 3:
                    pass
                elif hi == 5 and defer_const:
                    slots.append(("valu", (op3, ctx["tmp2"], val_vec, hash_vec_consts3[hi])))
                    slots.append(("valu", ("^", val_vec, val_vec, ctx["tmp2"])))
                else:
                    mul_vec = hash_mul_vecs[hi]
                    if mul_vec:
                        slots.append(("valu", ("multiply_add", val_vec, val_vec, mul_vec, hash_vec_consts1[hi])))
                    else:
                        slots.append(("valu", (op1, ctx["tmp1"], val_vec, hash_vec_consts1[hi])))
                        slots.append(("valu", (op3, ctx["tmp2"], val_vec, hash_vec_consts3[hi])))
                        slots.append(("valu", (op2, val_vec, ctx["tmp1"], ctx["tmp2"])))

        def emit_index_update(ctx, idx_vec, val_vec, level, invert_parity=False):
            if level == forest_height:
                slots.append(("valu", ("^", idx_vec, idx_vec, idx_vec)))
            else:
                for lane in range(VLEN):
                    slots.append(("alu", ("&", ctx["tmp1"] + lane, val_vec + lane, one_const)))
                    if invert_parity:
                        slots.append(("alu", ("-", ctx["node"] + lane, two_const, ctx["tmp1"] + lane)))
                    else:
                        slots.append(("alu", ("+", ctx["node"] + lane, ctx["tmp1"] + lane, one_const)))
                slots.append(("valu", ("multiply_add", idx_vec, idx_vec, two_vec, ctx["node"])))

        num_groups = (blocks_per_round + group_size - 1) // group_size
        num_round_tiles = (rounds + round_tile - 1) // round_tile

        for rt_idx in range(num_round_tiles):
            round_start = rt_idx * round_tile
            round_end = min(rounds, round_start + round_tile)
            is_last_tile = (rt_idx == num_round_tiles - 1)

            for group_idx in range(num_groups):
                group_start = group_idx * group_size
                active_blocks = min(group_size, blocks_per_round - group_start)

                for mini_start in range(0, active_blocks, mini_batch):
                    mini_end = min(mini_start + mini_batch, active_blocks)

                    for rnd in range(round_start, round_end):
                        level = rnd % (forest_height + 1)
                        defer_const = (rnd + 1 < rounds) and (((rnd + 1) % (forest_height + 1)) <= 3)
                        use_xor_nodes = rnd > 0
                        is_last_round = is_last_tile and (rnd == round_end - 1)

                        for gi in range(mini_start, mini_end):
                            block = group_start + gi
                            ctx = contexts[gi]
                            idx_vec = idx_base + block * VLEN
                            val_vec = val_base + block * VLEN
                            emit_tree_lookup(ctx, idx_vec, val_vec, level, use_xor_nodes)
                            emit_hash(ctx, val_vec, defer_const)
                            emit_index_update(ctx, idx_vec, val_vec, level, invert_parity=defer_const)

                            if is_last_round:
                                if block == 0:
                                    slots.append(("load", ("const", tmp_addr, INP_VALUES_P)))
                                    slots.append(("load", ("const", tmp_addr2, INP_INDICES_P)))
                                slots.append(("store", ("vstore", tmp_addr, val_vec)))
                                slots.append(("store", ("vstore", tmp_addr2, idx_vec)))
                                if block < blocks_per_round - 1:
                                    slots.append(("alu", ("+", tmp_addr, tmp_addr, vlen_const)))
                                    slots.append(("alu", ("+", tmp_addr2, tmp_addr2, vlen_const)))

        self.instrs.extend(_schedule_slots(slots))


if __name__ == "__main__":
    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)
    print(f"Scratch: {kb.scratch_ptr}, Cycles: {len(kb.instrs)}")
