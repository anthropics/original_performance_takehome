"""Analyze instruction mix to find optimization opportunities"""

from perf_takehome import KernelBuilder

def analyze_kernel_instructions():
    """Count instructions by type"""
    kb = KernelBuilder()
    kb.build_kernel(forest_height=10, n_nodes=1024, batch_size=256, rounds=16)

    engine_counts = {"alu": 0, "valu": 0, "load": 0, "store": 0, "flow": 0, "debug": 0}
    slot_counts = {"alu": 0, "valu": 0, "load": 0, "store": 0, "flow": 0, "debug": 0}

    for instr in kb.instrs:
        for engine, slots in instr.items():
            engine_counts[engine] += 1
            slot_counts[engine] += len(slots)

    total_instrs = len(kb.instrs)

    print(f"Total instructions (VLIW bundles): {total_instrs}")
    print(f"\nInstructions by engine:")
    for engine in ["alu", "valu", "load", "store", "flow", "debug"]:
        bundles = engine_counts[engine]
        slots = slot_counts[engine]
        if bundles > 0:
            avg_slots = slots / bundles
            print(f"  {engine:6s}: {bundles:5d} bundles, {slots:6d} total slots, {avg_slots:.2f} avg slots/bundle")

    print(f"\nTotal slots used:")
    for engine in ["alu", "valu", "load", "store", "flow"]:
        slots = slot_counts[engine]
        limit = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}[engine]
        utilization = (slots / (total_instrs * limit)) * 100 if total_instrs > 0 else 0
        print(f"  {engine:6s}: {slots:6d} / {total_instrs * limit:6d} max ({utilization:5.1f}% utilization)")

    # Look for patterns
    print(f"\nFlow operations breakdown:")
    flow_ops = {}
    for instr in kb.instrs:
        if "flow" in instr:
            for op in instr["flow"]:
                op_name = op[0] if isinstance(op, tuple) else op
                flow_ops[op_name] = flow_ops.get(op_name, 0) + 1

    for op_name, count in sorted(flow_ops.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_name:15s}: {count:5d}")

if __name__ == "__main__":
    analyze_kernel_instructions()
