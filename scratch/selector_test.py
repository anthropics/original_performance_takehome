import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from perf_takehome import KernelBuilder
from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
    N_CORES,
    VLEN,
    SCRATCH_SIZE,
)


class SelectorKernelBuilder(KernelBuilder):
    def __init__(self, selector_mode: str):
        super().__init__()
        self.selector_mode = selector_mode

    def _selector_round2(self, body, vec_idx, vec_val, vec_node, vec_tmp1, vec_tmp2,
                         vec_forest3, vec_forest4, vec_forest5, vec_forest6,
                         vec_two, vec_const2):
        """
        Selector for round 2 (idx in {3,4,5,6}).
        selector_mode:
          - "gather": use gather loads (no selector)
          - "chain": exact vselect chain by equality (slow)
          - "pair": two-level selector using idx>>2 and idx&1
        """
        if self.selector_mode == "gather":
            return False  # fall back to gather
        if self.selector_mode == "chain":
            # exact: start with 3, override with 4/5/6
            body.append(("valu", ("+", vec_node, vec_forest3, self.scratch_vconst(0))))
            for v, vf in ((4, vec_forest4), (5, vec_forest5), (6, vec_forest6)):
                body.append(("valu", ("==", vec_tmp1, vec_idx, self.scratch_vconst(v))))
                body.append(("flow", ("vselect", vec_node, vec_tmp1, vf, vec_node)))
            return True
        if self.selector_mode == "pair":
            # group = idx >> 2 (0 for 3/4, 1 for 5/6)
            # sel = idx & 1 (1 for odd => 3/5, 0 for even => 4/6)
            sub3 = self.scratch_vconst(0xFFFFFFFD)
            one = self.scratch_vconst(1)
            body.append(("valu", ("+", vec_tmp2, vec_idx, sub3)))
            body.append(("valu", (">>", vec_tmp2, vec_tmp2, one)))  # group (0 or 1)
            body.append(("valu", ("%", vec_tmp1, vec_idx, vec_two)))  # sel
            # pair0 = sel ? forest3 : forest4
            body.append(("flow", ("vselect", vec_node, vec_tmp1, vec_forest3, vec_forest4)))
            # pair1 = sel ? forest5 : forest6
            body.append(("flow", ("vselect", vec_tmp1, vec_tmp1, vec_forest5, vec_forest6)))
            # group select
            body.append(("flow", ("vselect", vec_node, vec_tmp2, vec_tmp1, vec_node)))
            return True
        return False

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        # Copy of current vector path with hooks for round-2 selector testing.
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

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
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []

        unroll = 8
        vec_tmp3 = self.alloc_scratch("vec_tmp3", length=VLEN)
        vec_idx_base = self.alloc_scratch("vec_idx_pool", length=unroll * VLEN)
        vec_val_base = self.alloc_scratch("vec_val_pool", length=unroll * VLEN)
        vec_node_base = self.alloc_scratch("vec_node_pool", length=unroll * VLEN)
        vec_tmp1_base = self.alloc_scratch("vec_tmp1_pool", length=unroll * VLEN)
        vec_tmp2_base = self.alloc_scratch("vec_tmp2_pool", length=unroll * VLEN)
        vec_addr_base = self.alloc_scratch("vec_addr_pool", length=unroll * VLEN)
        vec_addr_idx_base = self.alloc_scratch("vec_addr_idx_pool", length=unroll)
        vec_addr_val_base = self.alloc_scratch("vec_addr_val_pool", length=unroll)

        vec_idx_u = [vec_idx_base + ui * VLEN for ui in range(unroll)]
        vec_val_u = [vec_val_base + ui * VLEN for ui in range(unroll)]
        vec_node_u = [vec_node_base + ui * VLEN for ui in range(unroll)]
        vec_tmp1_u = [vec_tmp1_base + ui * VLEN for ui in range(unroll)]
        vec_tmp2_u = [vec_tmp2_base + ui * VLEN for ui in range(unroll)]
        vec_addr_u = [vec_addr_base + ui * VLEN for ui in range(unroll)]
        vec_addr_idx_u = [vec_addr_idx_base + ui for ui in range(unroll)]
        vec_addr_val_u = [vec_addr_val_base + ui for ui in range(unroll)]

        vec_zero = self.scratch_vconst(0)
        vec_one = self.scratch_vconst(1)
        vec_two = self.scratch_vconst(2)
        vec_const2 = self.scratch_vconst(2)
        vec_n_nodes = self.scratch_vbroadcast(self.scratch["n_nodes"])
        vec_forest_base = self.scratch_vbroadcast(self.scratch["forest_values_p"])

        # Small gather round 0/1 base constants
        forest0 = self.alloc_scratch("forest0")
        forest1 = self.alloc_scratch("forest1")
        forest2 = self.alloc_scratch("forest2")
        vec_forest0 = self.alloc_scratch("vec_forest0", length=VLEN)
        vec_forest1 = self.alloc_scratch("vec_forest1", length=VLEN)
        vec_forest2 = self.alloc_scratch("vec_forest2", length=VLEN)
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], zero_const)))
        body.append(("load", ("load", forest0, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest0, forest0)))
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], one_const)))
        body.append(("load", ("load", forest1, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest1, forest1)))
        body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], two_const)))
        body.append(("load", ("load", forest2, tmp_addr)))
        body.append(("valu", ("vbroadcast", vec_forest2, forest2)))

        # Preload 3..6 for selectors
        forest3 = self.alloc_scratch("forest3")
        forest4 = self.alloc_scratch("forest4")
        forest5 = self.alloc_scratch("forest5")
        forest6 = self.alloc_scratch("forest6")
        vec_forest3 = self.alloc_scratch("vec_forest3", length=VLEN)
        vec_forest4 = self.alloc_scratch("vec_forest4", length=VLEN)
        vec_forest5 = self.alloc_scratch("vec_forest5", length=VLEN)
        vec_forest6 = self.alloc_scratch("vec_forest6", length=VLEN)
        for v, fv, vfv in ((3, forest3, vec_forest3), (4, forest4, vec_forest4),
                           (5, forest5, vec_forest5), (6, forest6, vec_forest6)):
            c = self.scratch_const(v)
            body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], c)))
            body.append(("load", ("load", fv, tmp_addr)))
            body.append(("valu", ("vbroadcast", vfv, fv)))

        for rnd in range(rounds):
            vec_limit = (batch_size // VLEN) * VLEN
            for base in range(0, vec_limit, VLEN * unroll):
                # address setup
                for ui in range(unroll):
                    base_u = base + ui * VLEN
                    if base_u >= vec_limit:
                        continue
                    base_u_const = self.scratch_const(base_u)
                    body.append(("alu", ("+", vec_addr_idx_u[ui], self.scratch["inp_indices_p"], base_u_const)))
                    body.append(("alu", ("+", vec_addr_val_u[ui], self.scratch["inp_values_p"], base_u_const)))

                # vload idx/val
                for ui in range(unroll):
                    base_u = base + ui * VLEN
                    if base_u >= vec_limit:
                        continue
                    body.append(("load", ("vload", vec_idx_u[ui], vec_addr_idx_u[ui])))
                    body.append(("load", ("vload", vec_val_u[ui], vec_addr_val_u[ui])))

                # gather or selector
                if rnd not in (0, 1, 2):
                    for ui in range(unroll):
                        base_u = base + ui * VLEN
                        if base_u >= vec_limit:
                            continue
                        body.append(("valu", ("+", vec_addr_u[ui], vec_forest_base, vec_idx_u[ui])))
                        for lane in range(VLEN):
                            body.append(("load", ("load_offset", vec_node_u[ui], vec_addr_u[ui], lane)))

                val_groups = []
                idx_vecs = []
                addr_idx_vecs = []
                addr_val_vecs = []
                for ui in range(unroll):
                    base_u = base + ui * VLEN
                    if base_u >= vec_limit:
                        continue
                    if rnd == 0:
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_forest0)))
                    elif rnd == 1:
                        body.append(("valu", ("%", vec_tmp1_u[ui], vec_idx_u[ui], vec_two)))
                        body.append(("flow", ("vselect", vec_node_u[ui], vec_tmp1_u[ui], vec_forest1, vec_forest2)))
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))
                    elif rnd == 2:
                        used = self._selector_round2(
                            body, vec_idx_u[ui], vec_val_u[ui], vec_node_u[ui],
                            vec_tmp1_u[ui], vec_tmp2_u[ui],
                            vec_forest3, vec_forest4, vec_forest5, vec_forest6,
                            vec_two, vec_const2
                        )
                        if not used:
                            body.append(("valu", ("+", vec_addr_u[ui], vec_forest_base, vec_idx_u[ui])))
                            for lane in range(VLEN):
                                body.append(("load", ("load_offset", vec_node_u[ui], vec_addr_u[ui], lane)))
                            body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))
                    else:
                        body.append(("valu", ("^", vec_val_u[ui], vec_val_u[ui], vec_node_u[ui])))

                    val_groups.append((vec_val_u[ui], vec_tmp1_u[ui], vec_tmp2_u[ui]))
                    idx_vecs.append(vec_idx_u[ui])
                    addr_idx_vecs.append(vec_addr_idx_u[ui])
                    addr_val_vecs.append(vec_addr_val_u[ui])

                body.extend(self.build_hash_vec_multi(val_groups, rnd, base))

                for vi, (val_vec, t1, _t2) in enumerate(val_groups):
                    idx_vec = idx_vecs[vi]
                    body.append(("valu", ("%", t1, val_vec, vec_two)))
                    body.append(("valu", ("==", t1, t1, vec_zero)))
                    body.append(("flow", ("vselect", vec_tmp3, t1, vec_one, vec_two)))
                    body.append(("valu", ("*", idx_vec, idx_vec, vec_two)))
                    body.append(("valu", ("+", idx_vec, idx_vec, vec_tmp3)))
                    body.append(("valu", ("<", t1, idx_vec, vec_n_nodes)))
                    body.append(("flow", ("vselect", idx_vec, t1, idx_vec, vec_zero)))
                    body.append(("store", ("vstore", addr_idx_vecs[vi], idx_vec)))
                    body.append(("store", ("vstore", addr_val_vecs[vi], val_vec)))

        self.instrs.extend(self.build(body, vliw=True))
        self.instrs.append({"flow": [("pause",)]})


def run_selector_test(mode: str):
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    kb = SelectorKernelBuilder(mode)
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass
    inp_values_p = ref_mem[6]
    ok = (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    )
    print(f"{mode}: cycles={machine.cycle} ok={ok}")


if __name__ == "__main__":
    for mode in ["gather", "chain", "pair"]:
        run_selector_test(mode)
