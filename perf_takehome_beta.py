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

    def add_bundle(
        self,
        *,
        alu: list[tuple] | None = None,
        valu: list[tuple] | None = None,
        load: list[tuple] | None = None,
        store: list[tuple] | None = None,
        flow: list[tuple] | None = None,
        debug: list[tuple] | None = None,
    ):
        instr = {}
        if alu:
            instr["alu"] = alu
        if valu:
            instr["valu"] = valu
        if load:
            instr["load"] = load
        if store:
            instr["store"] = store
        if flow:
            instr["flow"] = flow
        if debug:
            instr["debug"] = debug
        for name, slots in instr.items():
            assert len(slots) <= SLOT_LIMITS[name]
        self.instrs.append(instr)

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
            self.add_bundle(load=[("const", addr, val)])
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round=None, i=None, debug=True):
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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Optimized implementation using VLIW packing and SIMD where possible.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_node = self.alloc_scratch("tmp_node")

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
            self.add_bundle(load=[("const", tmp1, i)])
            self.add_bundle(load=[("load", self.scratch[v], tmp1)])

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        vlen_const = self.scratch_const(VLEN)

        idx_base = self.alloc_scratch("idx", batch_size)
        val_base = self.alloc_scratch("val", batch_size)
        idx_ptr = self.alloc_scratch("idx_ptr")
        val_ptr = self.alloc_scratch("val_ptr")

        node_buf0 = self.alloc_scratch("node0", VLEN)
        node_buf1 = self.alloc_scratch("node1", VLEN)
        addr_buf0 = self.alloc_scratch("addr0", VLEN)
        addr_buf1 = self.alloc_scratch("addr1", VLEN)
        tmpv1 = self.alloc_scratch("tmpv1", VLEN)
        tmpv2 = self.alloc_scratch("tmpv2", VLEN)
        idx2_vec = self.alloc_scratch("idx2", VLEN)

        hash_consts = {0, 1}
        for _, val1, _, _, val3 in HASH_STAGES:
            hash_consts.add(val1)
            hash_consts.add(val3)

        const_scalars = {}
        for val in sorted(hash_consts):
            const_scalars[val] = self.scratch_const(val)

        const_vecs = {}
        vbroadcasts = []
        for val in sorted(hash_consts):
            dest = self.alloc_scratch(f"vconst_{val}", VLEN)
            const_vecs[val] = dest
            vbroadcasts.append(("vbroadcast", dest, const_scalars[val]))

        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        vbroadcasts.append(("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        for i in range(0, len(vbroadcasts), SLOT_LIMITS["valu"]):
            chunk = vbroadcasts[i : i + SLOT_LIMITS["valu"]]
            pause = i + SLOT_LIMITS["valu"] >= len(vbroadcasts)
            self.add_bundle(valu=chunk, flow=[("pause",)] if pause else None)

        num_blocks = batch_size // VLEN
        tail = batch_size % VLEN

        def addr_slots(idx_addr, addr_buf):
            return [
                ("+", addr_buf + lane, self.scratch["forest_values_p"], idx_addr + lane)
                for lane in range(VLEN)
            ]

        def load_slots(node_buf, addr_buf):
            return [
                ("load_offset", node_buf, addr_buf, lane) for lane in range(VLEN)
            ]

        def block_cycles(val_addr, idx_addr, node_buf):
            cycles = []
            cycles.append(
                {
                    "valu": [
                        ("^", val_addr, val_addr, node_buf),
                        ("+", idx2_vec, idx_addr, idx_addr),
                    ]
                }
            )
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                cycles.append(
                    {
                        "valu": [
                            (op1, tmpv1, val_addr, const_vecs[val1]),
                            (op3, tmpv2, val_addr, const_vecs[val3]),
                        ]
                    }
                )
                cycles.append({"valu": [(op2, val_addr, tmpv1, tmpv2)]})
            cycles.append({"valu": [("&", tmpv1, val_addr, const_vecs[1])]})
            cycles.append({"valu": [("+", tmpv1, tmpv1, const_vecs[1])]})
            cycles.append({"valu": [("+", idx_addr, idx2_vec, tmpv1)]})
            cycles.append({"valu": [("<", tmpv2, idx_addr, n_nodes_vec)]})
            cycles.append({"flow": [("vselect", idx_addr, tmpv2, idx_addr, const_vecs[0])]})
            return cycles

        self.add_bundle(
            alu=[
                ("+", idx_ptr, self.scratch["inp_indices_p"], zero_const),
                ("+", val_ptr, self.scratch["inp_values_p"], zero_const),
            ]
        )
        for b in range(num_blocks):
            offset = b * VLEN
            self.add_bundle(
                load=[
                    ("vload", idx_base + offset, idx_ptr),
                    ("vload", val_base + offset, val_ptr),
                ],
                alu=[("+", val_ptr, val_ptr, vlen_const)],
                flow=[("add_imm", idx_ptr, idx_ptr, VLEN)],
            )
        for r in range(tail):
            offset = num_blocks * VLEN + r
            self.add_bundle(
                load=[
                    ("load", idx_base + offset, idx_ptr),
                    ("load", val_base + offset, val_ptr),
                ],
                alu=[("+", val_ptr, val_ptr, one_const)],
                flow=[("add_imm", idx_ptr, idx_ptr, 1)],
            )

        for _ in range(rounds):
            if num_blocks:
                self.add_bundle(alu=addr_slots(idx_base, addr_buf0))
                loads = load_slots(node_buf0, addr_buf0)
                for li in range(0, VLEN, 2):
                    self.add_bundle(load=loads[li : li + 2])

                for b in range(num_blocks):
                    cur_idx = idx_base + b * VLEN
                    cur_val = val_base + b * VLEN
                    cur_node = node_buf0 if b % 2 == 0 else node_buf1
                    cycles = block_cycles(cur_val, cur_idx, cur_node)

                    if b + 1 < num_blocks:
                        next_idx = idx_base + (b + 1) * VLEN
                        next_addr = addr_buf1 if b % 2 == 0 else addr_buf0
                        next_node = node_buf1 if b % 2 == 0 else node_buf0
                        cycles[0].setdefault("alu", []).extend(
                            addr_slots(next_idx, next_addr)
                        )
                        next_loads = load_slots(next_node, next_addr)
                        for li, slot in enumerate(next_loads):
                            ci = 1 + (li // 2)
                            cycles[ci].setdefault("load", []).append(slot)

                    for cyc in cycles:
                        self.add_bundle(**cyc)

            for i in range(num_blocks * VLEN, batch_size):
                idx_addr = idx_base + i
                val_addr = val_base + i
                self.add_bundle(
                    alu=[("+", tmp_addr, self.scratch["forest_values_p"], idx_addr)]
                )
                self.add_bundle(load=[("load", tmp_node, tmp_addr)])
                self.add_bundle(alu=[("^", val_addr, val_addr, tmp_node)])
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    self.add_bundle(
                        alu=[(op1, tmp1, val_addr, self.scratch_const(val1))]
                    )
                    self.add_bundle(
                        alu=[(op3, tmp2, val_addr, self.scratch_const(val3))]
                    )
                    self.add_bundle(alu=[(op2, val_addr, tmp1, tmp2)])
                self.add_bundle(alu=[("&", tmp1, val_addr, one_const)])
                self.add_bundle(alu=[("+", tmp1, tmp1, one_const)])
                self.add_bundle(alu=[("+", tmp2, idx_addr, idx_addr)])
                self.add_bundle(alu=[("+", idx_addr, tmp2, tmp1)])
                self.add_bundle(
                    alu=[("<", tmp1, idx_addr, self.scratch["n_nodes"])]
                )
                self.add_bundle(
                    flow=[("select", idx_addr, tmp1, idx_addr, zero_const)]
                )

        self.add_bundle(
            alu=[
                ("+", idx_ptr, self.scratch["inp_indices_p"], zero_const),
                ("+", val_ptr, self.scratch["inp_values_p"], zero_const),
            ]
        )
        for b in range(num_blocks):
            offset = b * VLEN
            is_last = b == num_blocks - 1 and tail == 0
            flow_slots = [("pause",)] if is_last else [("add_imm", idx_ptr, idx_ptr, VLEN)]
            alu_slots = [] if is_last else [("+", val_ptr, val_ptr, vlen_const)]
            self.add_bundle(
                store=[
                    ("vstore", idx_ptr, idx_base + offset),
                    ("vstore", val_ptr, val_base + offset),
                ],
                alu=alu_slots,
                flow=flow_slots,
            )
        for r in range(tail):
            offset = num_blocks * VLEN + r
            is_last = r == tail - 1
            flow_slots = [("pause",)] if is_last else [("add_imm", idx_ptr, idx_ptr, 1)]
            alu_slots = [] if is_last else [("+", val_ptr, val_ptr, one_const)]
            self.add_bundle(
                store=[
                    ("store", idx_ptr, idx_base + offset),
                    ("store", val_ptr, val_base + offset),
                ],
                alu=alu_slots,
                flow=flow_slots,
            )

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
