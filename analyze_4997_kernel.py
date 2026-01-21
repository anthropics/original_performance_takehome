"""
Analyze the 4,997-cycle kernel to find remaining optimization opportunities
"""

from perf_takehome_vselect_packed import VSelectPackedKernel
from problem import SLOT_LIMITS

kb = VSelectPackedKernel()
kb.build_kernel(10, 2047, 256, 16)

print(f"Total bundles: {len(kb.instrs)}")

# Count operations by type
op_counts = {eng: 0 for eng in SLOT_LIMITS}
for bundle in kb.instrs:
    for eng, ops in bundle.items():
        if eng in SLOT_LIMITS:
            op_counts[eng] += len(ops)

print("\nOperation counts:")
for eng in ['alu', 'valu', 'load', 'store', 'flow']:
    print(f"  {eng:6s}: {op_counts[eng]:6d} ops")

# Estimate cycles by operation type (rough)
# Assuming most bundles have one primary operation type
alu_bundles = sum(1 for b in kb.instrs if 'alu' in b and b['alu'])
valu_bundles = sum(1 for b in kb.instrs if 'valu' in b and b['valu'])
load_bundles = sum(1 for b in kb.instrs if 'load' in b and b['load'])
store_bundles = sum(1 for b in kb.instrs if 'store' in b and b['store'])
flow_bundles = sum(1 for b in kb.instrs if 'flow' in b and b['flow'])

print(f"\nBundle counts by primary type:")
print(f"  ALU bundles:   {alu_bundles}")
print(f"  VALU bundles:  {valu_bundles}")
print(f"  Load bundles:  {load_bundles}")
print(f"  Store bundles: {store_bundles}")
print(f"  Flow bundles:  {flow_bundles}")

# Calculate theoretical minimums
n_elements = 256
n_rounds = 16

# Each element needs: load idx, load val (start), store idx, store val (end) = 4 mem ops
# But we vectorize 8x, so 256/8 = 32 vectors × 4 = 128 memory bundles (loads/stores at boundaries)
theoretical_boundary_mem = 32 * 2 * 2  # 32 vectors × (load idx + load val) × 2 (start + end)

# Per round: each of 256 elements needs 1 node load = 256 loads
# Vectorized: 256/8 = 32 vload ops, but we pack 2 per bundle = 16 bundles
# But wait, we process 48 elements (6 vectors) at a time, and do scalar loads
# 48 elements × 16 rounds = 768 loads per batch
# We have 256/48 = 5.33 batches, so ~6 batches
# 768 loads × 6 batches / 2 (pack 2 per bundle) = 2,304 load bundles

theoretical_node_loads = (256 * 16) / 2  # 2 loads per bundle

print(f"\nTheoretical minimums:")
print(f"  Boundary mem ops: {theoretical_boundary_mem} bundles")
print(f"  Node loads: {theoretical_node_loads} bundles")
print(f"  Total: {theoretical_boundary_mem + theoretical_node_loads} bundles")

# In reality
actual_load_bundles = load_bundles
print(f"\nActual load bundles: {actual_load_bundles}")
print(f"Overhead: {actual_load_bundles - theoretical_boundary_mem - theoretical_node_loads} bundles")

# If we could deduplicate 36.5% of node loads:
deduplicated_node_loads = theoretical_node_loads * 0.635  # Keep only 63.5%
savings = theoretical_node_loads - deduplicated_node_loads
new_total = theoretical_boundary_mem + deduplicated_node_loads

print(f"\nWith 36.5% deduplication:")
print(f"  Node loads: {deduplicated_node_loads} bundles")
print(f"  Savings: {savings} bundles")
print(f"  New total: {new_total} bundles")
print(f"  Projected cycles: ~{4997 - savings:.0f} cycles")

# In converged rounds (4 out of 16 rounds), 70% sharing
converged_rounds = 4
normal_rounds = 12
converged_node_loads = (256 * converged_rounds) / 2 * 0.30  # Keep only 30%
normal_node_loads = (256 * normal_rounds) / 2 * 0.635  # Keep 63.5%
mixed_total = theoretical_boundary_mem + converged_node_loads + normal_node_loads
mixed_savings = theoretical_node_loads - (converged_node_loads + normal_node_loads)

print(f"\nWith focused deduplication (70% in 4 converged rounds, 36.5% in others):")
print(f"  Converged round loads: {converged_node_loads} bundles")
print(f"  Normal round loads: {normal_node_loads} bundles")
print(f"  Total node loads: {converged_node_loads + normal_node_loads} bundles")
print(f"  Savings: {mixed_savings} bundles")
print(f"  Projected cycles: ~{4997 - mixed_savings:.0f} cycles")

if 4997 - mixed_savings < 1487:
    print(f"\n✓✓✓ THIS WOULD BEAT OPUS 4.5 TARGET!")
else:
    shortfall = (4997 - mixed_savings) - 1487
    print(f"\nStill {shortfall:.0f} cycles short of target")
    print("Need additional optimizations beyond node deduplication")
