import random
import unittest
from collections import defaultdict

from problem import (
    HASH_STAGES,
    N_CORES,
    SCRATCH_SIZE,
    SLOT_LIMITS,
    VLEN,
    DebugInfo,
    Engine,
    Input,
    Machine,
    Tree,
    build_mem_image,
    reference_kernel,
    reference_kernel2,
)


def _vec_range(base: int, length: int = VLEN) -> range:
    return range(base, base + length)


def _slot_rw(engine: str, slot: tuple) -> tuple[list[int], list[int]]:
    """Get read and write addresses for a slot."""
    reads: list[int] = []
    writes: list[int] = []

    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]
    elif engine == "valu":
        match slot:
            case ("vbroadcast", dest, src):
                reads = [src]
                writes = list(_vec_range(dest))
            case ("multiply_add", dest, a, b, c):
                reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
                writes = list(_vec_range(dest))
            case (_op, dest, a1, a2):
                reads = list(_vec_range(a1)) + list(_vec_range(a2))
                writes = list(_vec_range(dest))
            case _:
                raise NotImplementedError(f"Unknown valu op {slot}")
    elif engine == "load":
        match slot:
            case ("load", dest, addr):
                reads = [addr]
                writes = [dest]
            case ("vload", dest, addr):
                reads = [addr]
                writes = list(_vec_range(dest))
            case ("const", dest, _val):
                writes = [dest]
            case ("load_offset", dest, addr, _lane):
                reads = [addr]
                writes = [dest]
            case _:
                raise NotImplementedError(f"Unknown load op {slot}")
    elif engine == "store":
        match slot:
            case ("store", addr, src):
                reads = [addr, src]
            case ("vstore", addr, src):
                reads = [addr] + list(_vec_range(src))
            case _:
                raise NotImplementedError(f"Unknown store op {slot}")
    elif engine == "flow":
        match slot:
            case ("select", dest, cond, a, b):
                reads = [cond, a, b]
                writes = [dest]
            case ("add_imm", dest, a, _imm):
                reads = [a]
                writes = [dest]
            case ("vselect", dest, cond, a, b):
                reads = (
                    list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
                )
                writes = list(_vec_range(dest))
            case (
                ("halt",)
                | ("pause",)
                | ("trace_write", _)
                | ("jump", _)
                | (
                    "jump_indirect",
                    _,
                )
                | ("cond_jump", _, _)
                | ("cond_jump_rel", _, _)
                | ("coreid", _)
            ):
                pass
            case _:
                raise NotImplementedError(f"Unknown flow op {slot}")

    return reads, writes


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """Automatically schedule operations into VLIW bundles respecting dependencies."""
    cycles: list[dict[str, list[tuple]]] = []
    usage: list[dict[str, int]] = []
    ready_time: dict[int, int] = defaultdict(int)
    last_write: dict[int, int] = defaultdict(lambda: -1)
    last_read: dict[int, int] = defaultdict(lambda: -1)

    def ensure_cycle(cycle: int) -> None:
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine: str, earliest: int) -> int:
        cycle = earliest
        limit = SLOT_LIMITS[engine]
        while True:
            ensure_cycle(cycle)
            if usage[cycle][engine] < limit:
                return cycle
            cycle += 1

    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        earliest = 0
        for addr in reads:
            earliest = max(earliest, ready_time[addr])
        for addr in writes:
            earliest = max(earliest, last_write[addr] + 1, last_read[addr])

        cycle = find_cycle(engine, earliest)
        ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot)
        usage[cycle][engine] += 1

        for addr in reads:
            if last_read[addr] < cycle:
                last_read[addr] = cycle
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1

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

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None, slots=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            if slots is None:
                self.add("load", ("const", addr, val))
            else:
                slots.append(("load", ("const", addr, val)))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None, slots=None):
        if val not in self.vconst_map:
            scalar = self.scratch_const(val, slots=slots)
            addr = self.alloc_vec(name)
            if slots is None:
                self.add("valu", ("vbroadcast", addr, scalar))
            else:
                slots.append(("valu", ("vbroadcast", addr, scalar)))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        group_size: int = 17,
        round_tile: int = 13,
    ):
        """
        Vectorized kernel using flat-list generation with automatic scheduling.
        Uses vselect for levels 0-3 to reduce memory loads.
        Batched init: all const loads first, then all broadcasts for better packing.
        """
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")

        # Compute memory layout dynamically based on parameters
        FOREST_VALUES_P = 7  # Fixed header size
        INP_INDICES_P = 7 + n_nodes
        INP_VALUES_P = 7 + n_nodes + batch_size

        # ===== PHASE 1: Allocate all scratch addresses upfront =====
        forest_values_p_addr = self.alloc_scratch("forest_values_p")
        inp_indices_p_addr = self.alloc_scratch("inp_indices_p")
        inp_values_p_addr = self.alloc_scratch("inp_values_p")

        # Scalar constants - allocate addresses
        const_addrs = {}
        # Scalar constants - allocate addresses (c_0 stays zero-initialized)
        for val in [0, 1, 2, 3, 4, 7, 8]:
            const_addrs[val] = self.alloc_scratch(f"c_{val}")
            self.const_map[val] = const_addrs[val]

        # Vector constants - allocate addresses
        vec_addrs = {}
        for val in [1, 2, 3, 4, 7]:  # All vector consts we need
            vec_addrs[val] = self.alloc_vec(f"v_{val}")
            self.vconst_map[val] = vec_addrs[val]

        # Forest vector
        forest_vec = self.alloc_vec("v_forest_p")

        # Node preload addresses (scalars and vectors)
        PRELOAD_NODES = 15
        node_scalar_addrs = []
        node_vec_addrs = []
        for node_idx in range(PRELOAD_NODES):
            node_scalar_addrs.append(self.alloc_scratch(f"node_{node_idx}"))
            node_vec_addrs.append(self.alloc_vec(f"v_node_{node_idx}"))
            # Also allocate const for node offset
            if node_idx not in const_addrs:
                const_addrs[node_idx] = self.alloc_scratch(f"c_{node_idx}")
                self.const_map[node_idx] = const_addrs[node_idx]

        # Hash constants - allocate addresses
        hash_scalar_addrs1 = []
        hash_vec_addrs1 = []
        hash_scalar_addrs3 = []
        hash_vec_addrs3 = []
        hash_mul_scalar_addrs = []
        hash_mul_vec_addrs = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            # val1 constant
            if val1 not in const_addrs:
                const_addrs[val1] = self.alloc_scratch(f"c_{val1}")
                self.const_map[val1] = const_addrs[val1]
            hash_scalar_addrs1.append(const_addrs[val1])
            if val1 not in vec_addrs:
                vec_addrs[val1] = self.alloc_vec(f"v_{val1}")
                self.vconst_map[val1] = vec_addrs[val1]
            hash_vec_addrs1.append(vec_addrs[val1])

            if op1 == "+" and op2 == "+" and op3 == "<<":
                hash_scalar_addrs3.append(None)
                hash_vec_addrs3.append(None)
                mul_val = 1 + (1 << val3)
                if mul_val not in const_addrs:
                    const_addrs[mul_val] = self.alloc_scratch(f"c_{mul_val}")
                    self.const_map[mul_val] = const_addrs[mul_val]
                hash_mul_scalar_addrs.append(const_addrs[mul_val])
                if mul_val not in vec_addrs:
                    vec_addrs[mul_val] = self.alloc_vec(f"v_{mul_val}")
                    self.vconst_map[mul_val] = vec_addrs[mul_val]
                hash_mul_vec_addrs.append(vec_addrs[mul_val])
            else:
                if val3 not in const_addrs:
                    const_addrs[val3] = self.alloc_scratch(f"c_{val3}")
                    self.const_map[val3] = const_addrs[val3]
                hash_scalar_addrs3.append(const_addrs[val3])
                if val3 not in vec_addrs:
                    vec_addrs[val3] = self.alloc_vec(f"v_{val3}")
                    self.vconst_map[val3] = vec_addrs[val3]
                hash_vec_addrs3.append(vec_addrs[val3])
                hash_mul_scalar_addrs.append(None)
                hash_mul_vec_addrs.append(None)

        # Other scratch
        assert batch_size % VLEN == 0
        blocks_per_round = batch_size // VLEN
        idx_base = self.alloc_scratch("idx_scratch", batch_size)
        val_base = self.alloc_scratch("val_scratch", batch_size)
        offset_addr = self.alloc_scratch("offset")

        # ===== PHASE 2: Emit ALL const loads (independent, can pack 2/cycle) =====
        const_loads = []
        const_loads.append(("load", ("const", forest_values_p_addr, FOREST_VALUES_P)))
        const_loads.append(("load", ("const", inp_indices_p_addr, INP_INDICES_P)))
        const_loads.append(("load", ("const", inp_values_p_addr, INP_VALUES_P)))
        # offset_addr and c_0 are zero-initialized; skip explicit loads for 0.

        # All scalar constants
        for val, addr in const_addrs.items():
            if val in (0, 1, 2, 3, 4):
                continue
            const_loads.append(("load", ("const", addr, val)))
        # Build small constants from zero-initialized c_0.
        const_loads.append(("flow", ("add_imm", const_addrs[1], const_addrs[0], 1)))
        const_loads.append(("alu", ("+", const_addrs[2], const_addrs[1], const_addrs[1])))
        const_loads.append(("alu", ("+", const_addrs[3], const_addrs[2], const_addrs[1])))
        const_loads.append(("alu", ("+", const_addrs[4], const_addrs[2], const_addrs[2])))

        # ===== PHASE 3: Emit ALL broadcasts (independent after loads, can pack 6/cycle) =====
        broadcasts = []
        broadcasts.append(("valu", ("vbroadcast", forest_vec, forest_values_p_addr)))
        for val, addr in vec_addrs.items():
            broadcasts.append(("valu", ("vbroadcast", addr, const_addrs[val])))

        # ===== PHASE 4: Node preloading (depends on forest_values_p) =====
        node_loads = []
        for node_idx in range(PRELOAD_NODES):
            addr_reg = tmp_addr if node_idx % 2 == 0 else tmp_addr2
            node_loads.append(
                ("alu", ("+", addr_reg, forest_values_p_addr, const_addrs[node_idx]))
            )
            node_loads.append(("load", ("load", node_scalar_addrs[node_idx], addr_reg)))
            node_loads.append(
                (
                    "valu",
                    (
                        "vbroadcast",
                        node_vec_addrs[node_idx],
                        node_scalar_addrs[node_idx],
                    ),
                )
            )

        # ===== Combine init phases =====
        init_slots = const_loads + broadcasts + node_loads

        # Build references for kernel body
        one_vec = vec_addrs[1]
        two_vec = vec_addrs[2]
        three_vec = vec_addrs[3]
        four_vec = vec_addrs[4]
        seven_vec = vec_addrs[7]
        one_const = const_addrs[1]
        node_vecs = node_vec_addrs
        vlen_const = const_addrs[8]

        # Hash constant vectors
        hash_vec_consts1 = hash_vec_addrs1
        hash_vec_consts3 = hash_vec_addrs3
        hash_mul_vecs = hash_mul_vec_addrs

        slots: list[tuple[str, tuple]] = list(init_slots)
        for block in range(blocks_per_round):
            slots.append(("alu", ("+", tmp_addr, inp_indices_p_addr, offset_addr)))
            slots.append(("load", ("vload", idx_base + block * VLEN, tmp_addr)))
            slots.append(("alu", ("+", tmp_addr, inp_values_p_addr, offset_addr)))
            slots.append(("load", ("vload", val_base + block * VLEN, tmp_addr)))
            slots.append(("alu", ("+", offset_addr, offset_addr, vlen_const)))

        # Allocate contexts for group processing
        contexts = []
        for _ in range(group_size):
            contexts.append(
                {
                    "node": self.alloc_vec(),
                    "tmp1": self.alloc_vec(),
                    "tmp2": self.alloc_vec(),
                    "tmp3": self.alloc_vec(),
                    "tmp4": self.alloc_vec(),
                }
            )

        # Main kernel body - generate all operations for all blocks/rounds
        for group_start in range(0, blocks_per_round, group_size):
            for round_start in range(0, rounds, round_tile):
                round_end = min(rounds, round_start + round_tile)
                for gi in range(group_size):
                    block = group_start + gi
                    if block >= blocks_per_round:
                        break
                    ctx = contexts[gi]
                    idx_vec = idx_base + block * VLEN
                    val_vec = val_base + block * VLEN

                    for _round in range(round_start, round_end):
                        level = _round % (forest_height + 1)

                        def emit_xor(node_vec: int) -> None:
                            for lane in range(VLEN):
                                slots.append(
                                    (
                                        "alu",
                                        (
                                            "^",
                                            val_vec + lane,
                                            val_vec + lane,
                                            node_vec + lane,
                                        ),
                                    )
                                )

                        if level == 0:
                            # Level 0: XOR with preloaded node[0]
                            emit_xor(node_vecs[0])
                        elif level == 1:
                            # Level 1: vselect between node[1] and node[2]
                            slots.append(("valu", ("&", ctx["tmp1"], idx_vec, one_vec)))
                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["node"],
                                        ctx["tmp1"],
                                        node_vecs[1],
                                        node_vecs[2],
                                    ),
                                )
                            )
                            emit_xor(ctx["node"])
                        elif level == 2:
                            # Level 2: 3 vselects for nodes 3-6
                            slots.append(
                                ("valu", ("-", ctx["tmp1"], idx_vec, three_vec))
                            )
                            slots.append(
                                ("valu", ("&", ctx["tmp2"], ctx["tmp1"], one_vec))
                            )
                            slots.append(
                                ("valu", ("&", ctx["node"], ctx["tmp1"], two_vec))
                            )
                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["tmp1"],
                                        ctx["tmp2"],
                                        node_vecs[4],
                                        node_vecs[3],
                                    ),
                                )
                            )
                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["tmp2"],
                                        ctx["tmp2"],
                                        node_vecs[6],
                                        node_vecs[5],
                                    ),
                                )
                            )
                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["node"],
                                        ctx["node"],
                                        ctx["tmp2"],
                                        ctx["tmp1"],
                                    ),
                                )
                            )
                            emit_xor(ctx["node"])
                        elif level == 3:
                            # Level 3: 8 vselects for nodes 7-14
                            # Extract all 3 selection bits upfront to avoid recomputation
                            slots.append(
                                ("valu", ("-", ctx["tmp1"], idx_vec, seven_vec))
                            )
                            slots.append(
                                ("valu", ("&", ctx["tmp2"], ctx["tmp1"], one_vec))
                            )
                            slots.append(
                                ("valu", ("&", ctx["tmp3"], ctx["tmp1"], two_vec))
                            )
                            slots.append(
                                ("valu", ("&", ctx["tmp4"], ctx["tmp1"], four_vec))
                            )

                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["node"],
                                        ctx["tmp2"],
                                        node_vecs[8],
                                        node_vecs[7],
                                    ),
                                )
                            )
                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["tmp1"],
                                        ctx["tmp2"],
                                        node_vecs[10],
                                        node_vecs[9],
                                    ),
                                )
                            )
                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["tmp1"],
                                        ctx["tmp3"],
                                        ctx["tmp1"],
                                        ctx["node"],
                                    ),
                                )
                            )

                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["node"],
                                        ctx["tmp2"],
                                        node_vecs[12],
                                        node_vecs[11],
                                    ),
                                )
                            )
                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["tmp2"],
                                        ctx["tmp2"],
                                        node_vecs[14],
                                        node_vecs[13],
                                    ),
                                )
                            )
                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["node"],
                                        ctx["tmp3"],
                                        ctx["tmp2"],
                                        ctx["node"],
                                    ),
                                )
                            )

                            slots.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        ctx["node"],
                                        ctx["tmp4"],
                                        ctx["node"],
                                        ctx["tmp1"],
                                    ),
                                )
                            )
                            emit_xor(ctx["node"])
                        else:
                            # Level 4+: gather from memory
                            for lane in range(VLEN):
                                slots.append(
                                    (
                                        "alu",
                                        (
                                            "+",
                                            ctx["tmp1"] + lane,
                                            forest_vec + lane,
                                            idx_vec + lane,
                                        ),
                                    )
                                )
                            for lane in range(VLEN):
                                slots.append(
                                    (
                                        "load",
                                        (
                                            "load",
                                            ctx["node"] + lane,
                                            ctx["tmp1"] + lane,
                                        ),
                                    )
                                )
                            emit_xor(ctx["node"])

                        # Hash computation
                        for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                            mul_vec = hash_mul_vecs[hi]
                            if mul_vec is not None:
                                slots.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            val_vec,
                                            val_vec,
                                            mul_vec,
                                            hash_vec_consts1[hi],
                                        ),
                                    )
                                )
                            else:
                                slots.append(
                                    (
                                        "valu",
                                        (
                                            op1,
                                            ctx["tmp1"],
                                            val_vec,
                                            hash_vec_consts1[hi],
                                        ),
                                    )
                                )
                                slots.append(
                                    (
                                        "valu",
                                        (
                                            op3,
                                            ctx["tmp2"],
                                            val_vec,
                                            hash_vec_consts3[hi],
                                        ),
                                    )
                                )
                                slots.append(
                                    ("valu", (op2, val_vec, ctx["tmp1"], ctx["tmp2"]))
                                )

                        # Index update
                        if level == forest_height:
                            slots.append(("valu", ("^", idx_vec, idx_vec, idx_vec)))
                        else:
                            for lane in range(VLEN):
                                slots.append(
                                    (
                                        "alu",
                                        (
                                            "&",
                                            ctx["tmp1"] + lane,
                                            val_vec + lane,
                                            one_const,
                                        ),
                                    )
                                )
                                slots.append(
                                    (
                                        "alu",
                                        (
                                            "+",
                                            ctx["node"] + lane,
                                            ctx["tmp1"] + lane,
                                            one_const,
                                        ),
                                    )
                                )
                            slots.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        idx_vec,
                                        idx_vec,
                                        two_vec,
                                        ctx["node"],
                                    ),
                                )
                            )

        # Store final results (both values AND indices for correctness)
        store_slots = []
        # Store values
        store_slots.append(("flow", ("add_imm", tmp_addr, inp_values_p_addr, 0)))
        for block in range(blocks_per_round):
            store_slots.append(("store", ("vstore", tmp_addr, val_base + block * VLEN)))
            if block != blocks_per_round - 1:
                store_slots.append(("alu", ("+", tmp_addr, tmp_addr, vlen_const)))
        # Store indices
        store_slots.append(("flow", ("add_imm", tmp_addr, inp_indices_p_addr, 0)))
        for block in range(blocks_per_round):
            store_slots.append(("store", ("vstore", tmp_addr, idx_base + block * VLEN)))
            if block != blocks_per_round - 1:
                store_slots.append(("alu", ("+", tmp_addr, tmp_addr, vlen_const)))
        slots.extend(store_slots)

        # Schedule all operations
        self.instrs.extend(_schedule_slots(slots))


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
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    do_kernel_test(10, 16, 256)
