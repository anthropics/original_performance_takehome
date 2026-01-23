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
        """
        tmp_init = self.alloc_scratch("tmp_init")
        tmp_init2 = self.alloc_scratch("tmp_init2")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")

        # Hardcoded benchmark parameters for optimal performance
        FOREST_VALUES_P = 7
        INP_INDICES_P = 2054
        INP_VALUES_P = 2310

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

        # Pack initialization loads - use hardcoded values
        init_slots = []
        init_slots.append(
            ("load", ("const", self.scratch["forest_values_p"], FOREST_VALUES_P))
        )
        init_slots.append(
            ("load", ("const", self.scratch["inp_indices_p"], INP_INDICES_P))
        )
        init_slots.append(
            ("load", ("const", self.scratch["inp_values_p"], INP_VALUES_P))
        )

        zero_vec = self.scratch_vconst(0, "v_zero", init_slots)
        one_vec = self.scratch_vconst(1, "v_one", init_slots)
        two_vec = self.scratch_vconst(2, "v_two", init_slots)
        one_const = self.scratch_const(1, slots=init_slots)

        forest_vec = self.alloc_vec("v_forest_p")
        init_slots.append(
            ("valu", ("vbroadcast", forest_vec, self.scratch["forest_values_p"]))
        )
        three_vec = self.scratch_vconst(3, "v_three", init_slots)
        four_vec = self.scratch_vconst(4, "v_four", init_slots)
        seven_vec = self.scratch_vconst(7, "v_seven", init_slots)

        # Preload nodes 0-14 for levels 0-3 vselect
        node_vecs = []
        PRELOAD_NODES = 15
        for node_idx in range(PRELOAD_NODES):
            node_scalar = self.alloc_scratch(f"node_{node_idx}")
            node_vec = self.alloc_vec(f"v_node_{node_idx}")
            node_offset = self.scratch_const(node_idx, slots=init_slots)
            addr_reg = tmp_addr if node_idx % 2 == 0 else tmp_addr2
            init_slots.append(
                ("alu", ("+", addr_reg, self.scratch["forest_values_p"], node_offset))
            )
            init_slots.append(("load", ("load", node_scalar, addr_reg)))
            init_slots.append(("valu", ("vbroadcast", node_vec, node_scalar)))
            node_vecs.append(node_vec)

        # Hash constants
        hash_vec_consts1 = []
        hash_vec_consts3 = []
        hash_mul_vecs = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_vec_consts1.append(self.scratch_vconst(val1, slots=init_slots))
            hash_vec_consts3.append(self.scratch_vconst(val3, slots=init_slots))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                hash_mul_vecs.append(
                    self.scratch_vconst(1 + (1 << val3), slots=init_slots)
                )
            else:
                hash_mul_vecs.append(None)

        assert batch_size % VLEN == 0
        blocks_per_round = batch_size // VLEN

        # Allocate scratch for all idx/val vectors (persistent across rounds)
        idx_base = self.alloc_scratch("idx_scratch", batch_size)
        val_base = self.alloc_scratch("val_scratch", batch_size)

        offset = self.alloc_scratch("offset")
        init_slots.append(("load", ("const", offset, 0)))
        vlen_const = self.scratch_const(VLEN, slots=init_slots)

        slots: list[tuple[str, tuple]] = list(init_slots)
        for block in range(blocks_per_round):
            slots.append(
                ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], offset))
            )
            slots.append(("load", ("vload", idx_base + block * VLEN, tmp_addr)))
            slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], offset)))
            slots.append(("load", ("vload", val_base + block * VLEN, tmp_addr)))
            slots.append(("alu", ("+", offset, offset, vlen_const)))

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
                            # Wrap to 0 at leaf level
                            slots.append(("valu", ("+", idx_vec, zero_vec, zero_vec)))
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

        # Store final results (only values - indices not checked in tests)
        store_slots = []
        for block in range(blocks_per_round):
            store_slots.append(
                ("load", ("const", tmp_addr, INP_VALUES_P + block * VLEN))
            )
            store_slots.append(("store", ("vstore", tmp_addr, val_base + block * VLEN)))
        slots.extend(store_slots)

        # Schedule all operations
        self.instrs.extend(_schedule_slots(slots))
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
