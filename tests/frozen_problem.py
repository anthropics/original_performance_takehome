"""
Read the top of perf_takehome.py for more introduction.

This file is separate mostly for ease of copying it to freeze the machine and
reference kernel for testing.
"""

from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
import random

Engine = Literal["alu", "load", "store", "flow"]
Instruction = dict[Engine, list[tuple]]


class CoreState(Enum):
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


@dataclass
class Core:
    id: int
    scratch: list[int]
    trace_buf: list[int]
    pc: int = 0
    state: CoreState = CoreState.RUNNING


@dataclass
class DebugInfo:
    """
    We give you some debug info but it's up to you to use it in Machine if you
    want to. You're also welcome to add more.
    """

    # Maps scratch variable addr to (name, len) pair
    scratch_map: dict[int, (str, int)]


def cdiv(a, b):
    return (a + b - 1) // b


SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
    "debug": 64,
}

VLEN = 8
# Older versions of the take-home used multiple cores, but this version only uses 1
N_CORES = 1
SCRATCH_SIZE = 1536
BASE_ADDR_TID = 100000


class Machine:
    """
    Simulator for a custom VLIW SIMD architecture.

    VLIW (Very Large Instruction Word): Cores are composed of different
    "engines" each of which can execute multiple "slots" per cycle in parallel.
    How many slots each engine can execute per cycle is limited by SLOT_LIMITS.
    Effects of instructions don't take effect until the end of cycle. Each
    cycle, all engines execute all of their filled slots for that instruction.
    Effects like writes to memory take place after all the inputs are read.

    SIMD: There are instructions for acting on vectors of VLEN elements in a
    single slot. You can use vload and vstore to load multiple contiguous
    elements but not non-contiguous elements. Use vbroadcast to broadcast a
    scalar to a vector and then operate on vectors with valu instructions.

    The memory and scratch space are composed of 32-bit words. The solution is
    plucked out of the memory at the end of the program. You can think of the
    scratch space as serving the purpose of registers, constant memory, and a
    manually-managed cache.

    Here's an example of what an instruction might look like:

    {"valu": [("*", 4, 0, 0), ("+", 8, 4, 0)], "load": [("load", 16, 17)]}

    In general every number in an instruction is a scratch address except for
    const and jump, and except for store and some flow instructions the first
    operand is the destination.

    This comment is not meant to be full ISA documentation though, for the rest
    you should look through the simulator code.
    """

    def __init__(
        self,
        mem_dump: list[int],
        program: list[Instruction],
        debug_info: DebugInfo,
        n_cores: int = 1,
        scratch_size: int = SCRATCH_SIZE,
        trace: bool = False,
        value_trace: dict[Any, int] = {},
    ):
        self.cores = [
            Core(id=i, scratch=[0] * scratch_size, trace_buf=[]) for i in range(n_cores)
        ]
        self.mem = copy(mem_dump)
        self.program = program
        self.debug_info = debug_info
        self.value_trace = value_trace
        self.prints = False
        self.cycle = 0
        self.enable_pause = True
        self.enable_debug = True
        if trace:
            self.setup_trace()
        else:
            self.trace = None

    def rewrite_instr(self, instr):
        """
        Rewrite an instruction to use scratch addresses instead of names
        """
        res = {}
        for name, slots in instr.items():
            res[name] = []
            for slot in slots:
                res[name].append(self.rewrite_slot(slot))
        return res

    def print_step(self, instr, core):
        # print(core.id)
        # print(core.trace_buf)
        print(self.scratch_map(core))
        print(core.pc, instr, self.rewrite_instr(instr))

    def scratch_map(self, core):
        res = {}
        for addr, (name, length) in self.debug_info.scratch_map.items():
            res[name] = core.scratch[addr : addr + length]
        return res

    def rewrite_slot(self, slot):
        return tuple(
            self.debug_info.scratch_map.get(s, (None, None))[0] or s for s in slot
        )

    def setup_trace(self):
        """
        The simulator generates traces in Chrome's Trace Event Format for
        visualization in Perfetto (or chrome://tracing if you prefer it). See
        the bottom of the file for info about how to use this.

        See the format docs in case you want to add more info to the trace:
        https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
        """
        self.trace = open("trace.json", "w")
        self.trace.write("[")
        tid_counter = 0
        self.tids = {}
        for ci, core in enumerate(self.cores):
            self.trace.write(
                f'{{"name": "process_name", "ph": "M", "pid": {ci}, "tid": 0, "args": {{"name":"Core {ci}"}}}},\n'
            )
            for name, limit in SLOT_LIMITS.items():
                if name == "debug":
                    continue
                for i in range(limit):
                    tid_counter += 1
                    self.trace.write(
                        f'{{"name": "thread_name", "ph": "M", "pid": {ci}, "tid": {tid_counter}, "args": {{"name":"{name}-{i}"}}}},\n'
                    )
                    self.tids[(ci, name, i)] = tid_counter

        # Add zero-length events at the start so all slots show up in Perfetto
        for ci, core in enumerate(self.cores):
            for name, limit in SLOT_LIMITS.items():
                if name == "debug":
                    continue
                for i in range(limit):
                    tid = self.tids[(ci, name, i)]
                    self.trace.write(
                        f'{{"name": "init", "cat": "op", "ph": "X", "pid": {ci}, "tid": {tid}, "ts": 0, "dur": 0}},\n'
                    )
        for ci, core in enumerate(self.cores):
            self.trace.write(
                f'{{"name": "process_name", "ph": "M", "pid": {len(self.cores) + ci}, "tid": 0, "args": {{"name":"Core {ci} Scratch"}}}},\n'
            )
            for addr, (name, length) in self.debug_info.scratch_map.items():
                self.trace.write(
                    f'{{"name": "thread_name", "ph": "M", "pid": {len(self.cores) + ci}, "tid": {BASE_ADDR_TID + addr}, "args": {{"name":"{name}-{length}"}}}},\n'
                )

    def run(self):
        for core in self.cores:
            if core.state == CoreState.PAUSED:
                core.state = CoreState.RUNNING
        while any(c.state == CoreState.RUNNING for c in self.cores):
            has_non_debug = False
            for core in self.cores:
                if core.state != CoreState.RUNNING:
                    continue
                if core.pc >= len(self.program):
                    core.state = CoreState.STOPPED
                    continue
                instr = self.program[core.pc]
                if self.prints:
                    self.print_step(instr, core)
                core.pc += 1
                self.step(instr, core)
                if any(name != "debug" for name in instr.keys()):
                    has_non_debug = True
            if has_non_debug:
                self.cycle += 1

    def alu(self, core, op, dest, a1, a2):
        a1 = core.scratch[a1]
        a2 = core.scratch[a2]
        match op:
            case "+":
                res = a1 + a2
            case "-":
                res = a1 - a2
            case "*":
                res = a1 * a2
            case "//":
                res = a1 // a2
            case "cdiv":
                res = cdiv(a1, a2)
            case "^":
                res = a1 ^ a2
            case "&":
                res = a1 & a2
            case "|":
                res = a1 | a2
            case "<<":
                res = a1 << a2
            case ">>":
                res = a1 >> a2
            case "%":
                res = a1 % a2
            case "<":
                res = int(a1 < a2)
            case "==":
                res = int(a1 == a2)
            case _:
                raise NotImplementedError(f"Unknown alu op {op}")
        res = res % (2**32)
        self.scratch_write[dest] = res

    def valu(self, core, *slot):
        match slot:
            case ("vbroadcast", dest, src):
                for i in range(VLEN):
                    self.scratch_write[dest + i] = core.scratch[src]
            case ("multiply_add", dest, a, b, c):
                for i in range(VLEN):
                    mul = (core.scratch[a + i] * core.scratch[b + i]) % (2**32)
                    self.scratch_write[dest + i] = (mul + core.scratch[c + i]) % (2**32)
            case (op, dest, a1, a2):
                for i in range(VLEN):
                    self.alu(core, op, dest + i, a1 + i, a2 + i)
            case _:
                raise NotImplementedError(f"Unknown valu op {slot}")

    def load(self, core, *slot):
        match slot:
            case ("load", dest, addr):
                # print(dest, addr, core.scratch[addr])
                self.scratch_write[dest] = self.mem[core.scratch[addr]]
            case ("load_offset", dest, addr, offset):
                # Handy for treating vector dest and addr as a full block in the mini-compiler if you want
                self.scratch_write[dest + offset] = self.mem[
                    core.scratch[addr + offset]
                ]
            case ("vload", dest, addr):  # addr is a scalar
                addr = core.scratch[addr]
                for vi in range(VLEN):
                    self.scratch_write[dest + vi] = self.mem[addr + vi]
            case ("const", dest, val):
                self.scratch_write[dest] = (val) % (2**32)
            case _:
                raise NotImplementedError(f"Unknown load op {slot}")

    def store(self, core, *slot):
        match slot:
            case ("store", addr, src):
                addr = core.scratch[addr]
                self.mem_write[addr] = core.scratch[src]
            case ("vstore", addr, src):  # addr is a scalar
                addr = core.scratch[addr]
                for vi in range(VLEN):
                    self.mem_write[addr + vi] = core.scratch[src + vi]
            case _:
                raise NotImplementedError(f"Unknown store op {slot}")

    def flow(self, core, *slot):
        match slot:
            case ("select", dest, cond, a, b):
                self.scratch_write[dest] = (
                    core.scratch[a] if core.scratch[cond] != 0 else core.scratch[b]
                )
            case ("add_imm", dest, a, imm):
                self.scratch_write[dest] = (core.scratch[a] + imm) % (2**32)
            case ("vselect", dest, cond, a, b):
                for vi in range(VLEN):
                    self.scratch_write[dest + vi] = (
                        core.scratch[a + vi]
                        if core.scratch[cond + vi] != 0
                        else core.scratch[b + vi]
                    )
            case ("halt",):
                core.state = CoreState.STOPPED
            case ("pause",):
                if self.enable_pause:
                    core.state = CoreState.PAUSED
            case ("trace_write", val):
                core.trace_buf.append(core.scratch[val])
            case ("cond_jump", cond, addr):
                if core.scratch[cond] != 0:
                    core.pc = addr
            case ("cond_jump_rel", cond, offset):
                if core.scratch[cond] != 0:
                    core.pc += offset
            case ("jump", addr):
                core.pc = addr
            case ("jump_indirect", addr):
                core.pc = core.scratch[addr]
            case ("coreid", dest):
                self.scratch_write[dest] = core.id
            case _:
                raise NotImplementedError(f"Unknown flow op {slot}")

    def trace_post_step(self, instr, core):
        # You can add extra stuff to the trace if you want!
        for addr, (name, length) in self.debug_info.scratch_map.items():
            if any((addr + vi) in self.scratch_write for vi in range(length)):
                val = str(core.scratch[addr : addr + length])
                val = val.replace("[", "").replace("]", "")
                self.trace.write(
                    f'{{"name": "{val}", "cat": "op", "ph": "X", "pid": {len(self.cores) + core.id}, "tid": {BASE_ADDR_TID + addr}, "ts": {self.cycle}, "dur": 1 }},\n'
                )

    def trace_slot(self, core, slot, name, i):
        self.trace.write(
            f'{{"name": "{slot[0]}", "cat": "op", "ph": "X", "pid": {core.id}, "tid": {self.tids[(core.id, name, i)]}, "ts": {self.cycle}, "dur": 1, "args":{{"slot": "{str(slot)}", "named": "{str(self.rewrite_slot(slot))}" }} }},\n'
        )

    def step(self, instr: Instruction, core):
        """
        Execute all the slots in each engine for a single instruction bundle
        """
        ENGINE_FNS = {
            "alu": self.alu,
            "valu": self.valu,
            "load": self.load,
            "store": self.store,
            "flow": self.flow,
        }
        self.scratch_write = {}
        self.mem_write = {}
        for name, slots in instr.items():
            if name == "debug":
                if not self.enable_debug:
                    continue
                for slot in slots:
                    if slot[0] == "compare":
                        loc, key = slot[1], slot[2]
                        ref = self.value_trace[key]
                        res = core.scratch[loc]
                        assert res == ref, f"{res} != {ref} for {key} at pc={core.pc}"
                    elif slot[0] == "vcompare":
                        loc, keys = slot[1], slot[2]
                        ref = [self.value_trace[key] for key in keys]
                        res = core.scratch[loc : loc + VLEN]
                        assert res == ref, (
                            f"{res} != {ref} for {keys} at pc={core.pc} loc={loc}"
                        )
                continue
            assert len(slots) <= SLOT_LIMITS[name]
            for i, slot in enumerate(slots):
                if self.trace is not None:
                    self.trace_slot(core, slot, name, i)
                ENGINE_FNS[name](core, *slot)
        for addr, val in self.scratch_write.items():
            core.scratch[addr] = val
        for addr, val in self.mem_write.items():
            self.mem[addr] = val

        if self.trace:
            self.trace_post_step(instr, core)

        del self.scratch_write
        del self.mem_write

    def __del__(self):
        if self.trace is not None:
            self.trace.write("]")
            self.trace.close()


@dataclass
class Tree:
    """
    An implicit perfect balanced binary tree with values on the nodes.
    """

    height: int
    values: list[int]

    @staticmethod
    def generate(height: int):
        n_nodes = 2 ** (height + 1) - 1
        values = [random.randint(0, 2**30 - 1) for _ in range(n_nodes)]
        return Tree(height, values)


@dataclass
class Input:
    """
    A batch of inputs, indices to nodes (starting as 0) and initial input
    values. We then iterate these for a specified number of rounds.
    """

    indices: list[int]
    values: list[int]
    rounds: int

    @staticmethod
    def generate(forest: Tree, batch_size: int, rounds: int):
        indices = [0 for _ in range(batch_size)]
        values = [random.randint(0, 2**30 - 1) for _ in range(batch_size)]
        return Input(indices, values, rounds)


HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]


def myhash(a: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for op1, val1, op2, op3, val3 in HASH_STAGES:
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))

    return a


def reference_kernel(t: Tree, inp: Input):
    """
    Reference implementation of the kernel.

    A parallel tree traversal where at each node we set
    cur_inp_val = myhash(cur_inp_val ^ node_val)
    and then choose the left branch if cur_inp_val is even.
    If we reach the bottom of the tree we wrap around to the top.
    """
    for h in range(inp.rounds):
        for i in range(len(inp.indices)):
            idx = inp.indices[i]
            val = inp.values[i]
            val = myhash(val ^ t.values[idx])
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= len(t.values) else idx
            inp.values[i] = val
            inp.indices[i] = idx


def build_mem_image(t: Tree, inp: Input) -> list[int]:
    """
    Build a flat memory image of the problem.
    """
    header = 7
    extra_room = len(t.values) + len(inp.indices) * 2 + VLEN * 2 + 32
    mem = [0] * (
        header + len(t.values) + len(inp.indices) + len(inp.values) + extra_room
    )
    forest_values_p = header
    inp_indices_p = forest_values_p + len(t.values)
    inp_values_p = inp_indices_p + len(inp.values)
    extra_room = inp_values_p + len(inp.values)

    mem[0] = inp.rounds
    mem[1] = len(t.values)
    mem[2] = len(inp.indices)
    mem[3] = t.height
    mem[4] = forest_values_p
    mem[5] = inp_indices_p
    mem[6] = inp_values_p
    mem[7] = extra_room

    mem[header:inp_indices_p] = t.values
    mem[inp_indices_p:inp_values_p] = inp.indices
    mem[inp_values_p:] = inp.values
    return mem


def myhash_traced(a: int, trace: dict[Any, int], round: int, batch_i: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))
        trace[(round, batch_i, "hash_stage", i)] = a

    return a


def reference_kernel2(mem: list[int], trace: dict[Any, int] = {}):
    """
    Reference implementation of the kernel on a flat memory.
    """
    # This is the initial memory layout
    rounds = mem[0]
    n_nodes = mem[1]
    batch_size = mem[2]
    forest_height = mem[3]
    # Offsets into the memory which indices get added to
    forest_values_p = mem[4]
    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    yield mem
    for h in range(rounds):
        for i in range(batch_size):
            idx = mem[inp_indices_p + i]
            trace[(h, i, "idx")] = idx
            val = mem[inp_values_p + i]
            trace[(h, i, "val")] = val
            node_val = mem[forest_values_p + idx]
            trace[(h, i, "node_val")] = node_val
            val = myhash_traced(val ^ node_val, trace, h, i)
            trace[(h, i, "hashed_val")] = val
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            trace[(h, i, "next_idx")] = idx
            idx = 0 if idx >= n_nodes else idx
            trace[(h, i, "wrapped_idx")] = idx
            mem[inp_values_p + i] = val
            mem[inp_indices_p + i] = idx
    # You can add new yields or move this around for debugging
    # as long as it's matched by pause instructions.
    # The submission tests evaluate only on final memory.
    yield mem
