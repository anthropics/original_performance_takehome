
import sys
import os

# Add parent directory to path to import problem.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perf_takehome import KernelBuilder, do_kernel_test, Machine, N_CORES, BASELINE, Tree, Input, build_mem_image, reference_kernel2, VLEN
import random
import unittest

# Patch KernelBuilder to accept BATCH as a parameter
def build_kernel_patched(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int, BATCH_SIZE=16):
    # Save original build_kernel method if needed, but we will overwrite the BATCH line dynamically
    # Actually, simpler to just inherit or modify the class in place?
    # Let's use the source code modification trick or just rewrite the class here slightly?
    # No, let's copy the FULL KernelBuilder class here to allow modification.
    
    # ... (Wait, copying the whole class is verbose. Better to modify 'perf_takehome.py' to accept an arg?)
    pass

# Strategy: We will create a clean "configurable" version of the kernel builder here.
# Copying relevant parts from perf_takehome.py and adding BATCH as arg.

from collections import defaultdict
from problem import SLOT_LIMITS, DebugInfo, HASH_STAGES, SCRATCH_SIZE

class ConfigurableKernelBuilder(KernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int, configured_batch=16):
        # ... (This needs the full body of build_kernel)
        # Since I can't easily import the *body* of the function, I will assume the user 
        # wants me to create a script that modifies the BATCH variable in the file and runs it.
        pass

if __name__ == "__main__":
    import subprocess
    import re
    
    batch_sizes = [6, 8, 12, 16, 20, 24]
    
    base_file = "perf_takehome.py"
    with open(base_file, 'r') as f:
        content = f.read()
        
    for b in batch_sizes:
        print(f"\n--- Testing BATCH = {b} ---")
        # Replace 'BATCH = 16' with 'BATCH = {b}'
        new_content = re.sub(r"BATCH = \d+", f"BATCH = {b}", content)
        
        with open("perf_temp.py", "w") as f:
            f.write(new_content)
            
        # Run the test
        try:
            result = subprocess.run(["python3", "perf_temp.py", "Tests.test_kernel_cycles"], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FAILED (Return Code {result.returncode})")
                print(result.stderr)
            else:
                # Extract CYCLES
                match = re.search(r"CYCLES:\s+(\d+)", result.stdout)
                if match:
                    print(f"CYCLES: {match.group(1)}")
                else:
                    print("Could not parse cycle count.")
        except Exception as e:
            print(f"Error: {e}")
            
    if os.path.exists("perf_temp.py"):
        os.remove("perf_temp.py")
