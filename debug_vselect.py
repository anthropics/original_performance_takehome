"""Debug the IndexError"""
from perf_takehome import KernelBuilder, HASH_STAGES, VLEN, do_kernel_test
import problem

# Patch the load function to print debug info
original_load = problem.Machine.load

def debug_load(self, core, *slot):
    if slot[0] == "load" and len(slot) == 3:
        dest, addr = slot[1], slot[2]
        addr_val = core.scratch[addr]
        if addr_val >= len(self.mem):
            print(f"ERROR: Trying to load from mem[{addr_val}] but mem size is {len(self.mem)}")
            print(f"  slot: {slot}")
            print(f"  addr register: {addr}, contains: {addr_val}")
            print(f"  dest register: {dest}")
            print(f"  cycle: {self.cycle}")
            print(f"  pc: {core.pc}")
            raise IndexError(f"mem index {addr_val} out of range")
    return original_load(self, core, *slot)

problem.Machine.load = debug_load

# Now import and test
from perf_takehome_vselect import VSelectCacheKernel
import perf_takehome

perf_takehome.KernelBuilder.build_kernel = VSelectCacheKernel.build_kernel
cycles = do_kernel_test(10, 16, 256)
print(f"Cycles: {cycles}")
