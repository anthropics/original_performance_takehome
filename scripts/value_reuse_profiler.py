#!/usr/bin/env python3
"""
Profiler to find value reuse opportunities in the kernel.
Tracks which scratch addresses are written and read, looking for:
1. Values that are computed multiple times (same computation, different addresses)
2. Values that are written then overwritten before being fully utilized
3. Recomputation patterns like the (idx-7) case we fixed
"""

import sys
sys.path.insert(0, '.')

from collections import defaultdict, Counter
from perf_takehome import KernelBuilder, _slot_rw, VLEN

def analyze_kernel():
    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)
    
    # Track all operations
    ops_by_pattern = defaultdict(list)  # pattern -> [(cycle, engine, slot), ...]
    writes_to_addr = defaultdict(list)  # addr -> [(cycle, engine, slot), ...]
    reads_from_addr = defaultdict(list)  # addr -> [(cycle, engine, slot), ...]
    
    # Track computation patterns (op, operand_pattern) -> count
    computation_patterns = Counter()
    
    # Analyze each instruction
    for cycle, instr in enumerate(kb.instrs):
        for engine, slots in instr.items():
            for slot in slots:
                reads, writes = _slot_rw(engine, slot)
                
                # Record reads and writes
                for addr in reads:
                    reads_from_addr[addr].append((cycle, engine, slot))
                for addr in writes:
                    writes_to_addr[addr].append((cycle, engine, slot))
                
                # Create computation pattern
                if engine in ('valu', 'alu') and len(slot) >= 3:
                    op = slot[0]
                    # Normalize operands to see if same computation repeated
                    if engine == 'valu':
                        # For VALU, track the operation type and relative operand positions
                        pattern = (engine, op, 'vec_op')
                    else:
                        pattern = (engine, op, 'scalar_op')
                    computation_patterns[pattern] += 1
                    ops_by_pattern[pattern].append((cycle, engine, slot))
    
    return kb, ops_by_pattern, writes_to_addr, reads_from_addr, computation_patterns

def find_recomputation_patterns(kb, writes_to_addr, reads_from_addr):
    """Find cases where similar computations happen multiple times"""
    
    print("=" * 70)
    print("RECOMPUTATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Group writes by the operation that produced them
    write_ops = defaultdict(list)  # (engine, op, src_pattern) -> [dest_addrs]
    
    for addr, writes in writes_to_addr.items():
        for cycle, engine, slot in writes:
            if engine in ('valu', 'alu') and len(slot) >= 3:
                op = slot[0]
                dest = slot[1]
                srcs = slot[2:]
                # Create a source pattern (relative to dest for vectors)
                src_pattern = tuple(s - dest if isinstance(s, int) else s for s in srcs)
                write_ops[(engine, op, src_pattern)].append((addr, cycle, slot))
    
    # Find patterns that appear multiple times
    print("Repeated computation patterns (same op + relative operands):")
    print("-" * 70)
    
    repeated = [(k, v) for k, v in write_ops.items() if len(v) > 100]
    repeated.sort(key=lambda x: -len(x[1]))
    
    for (engine, op, src_pattern), instances in repeated[:20]:
        print(f"\n{engine}.{op} with src_pattern {src_pattern}: {len(instances)} instances")
        # Show first few
        for addr, cycle, slot in instances[:3]:
            print(f"    cycle {cycle}: {slot}")
        if len(instances) > 3:
            print(f"    ... and {len(instances) - 3} more")

def find_overwritten_values(kb, writes_to_addr, reads_from_addr):
    """Find values that are written then overwritten with few reads in between"""
    
    print()
    print("=" * 70)
    print("OVERWRITTEN VALUE ANALYSIS")
    print("=" * 70)
    print()
    
    wasted_writes = []
    
    for addr, writes in writes_to_addr.items():
        if len(writes) < 2:
            continue
        
        reads = reads_from_addr.get(addr, [])
        read_cycles = set(r[0] for r in reads)
        
        # Check each pair of consecutive writes
        for i in range(len(writes) - 1):
            w1_cycle = writes[i][0]
            w2_cycle = writes[i + 1][0]
            
            # Count reads between these writes
            reads_between = sum(1 for rc in read_cycles if w1_cycle < rc < w2_cycle)
            
            if reads_between == 0:
                # Value written but never read before overwrite!
                wasted_writes.append((addr, writes[i], writes[i + 1], reads_between))
            elif reads_between == 1:
                # Only read once - might be opportunity for fusion
                wasted_writes.append((addr, writes[i], writes[i + 1], reads_between))
    
    # Group by pattern
    print(f"Found {len(wasted_writes)} potentially wasted/underutilized writes")
    print()
    
    # Show some examples
    print("Examples of writes that are quickly overwritten:")
    print("-" * 70)
    
    for addr, w1, w2, reads_between in wasted_writes[:10]:
        w1_cycle, w1_engine, w1_slot = w1
        w2_cycle, w2_engine, w2_slot = w2
        gap = w2_cycle - w1_cycle
        print(f"\nAddr {addr}: written at cycle {w1_cycle}, overwritten at cycle {w2_cycle} (gap={gap}, reads={reads_between})")
        print(f"    Write 1: {w1_engine} {w1_slot}")
        print(f"    Write 2: {w2_engine} {w2_slot}")

def analyze_hot_addresses(kb, writes_to_addr, reads_from_addr):
    """Find the most frequently accessed addresses"""
    
    print()
    print("=" * 70)
    print("HOT ADDRESS ANALYSIS")
    print("=" * 70)
    print()
    
    # Combine reads and writes
    access_counts = Counter()
    for addr, writes in writes_to_addr.items():
        access_counts[addr] += len(writes)
    for addr, reads in reads_from_addr.items():
        access_counts[addr] += len(reads)
    
    # Get address names from debug info
    addr_names = {v: k for k, v in kb.scratch.items()}
    
    print("Top 30 most accessed addresses:")
    print("-" * 70)
    print(f"{'Address':<10} {'Name':<25} {'Writes':<10} {'Reads':<10} {'Total':<10}")
    print("-" * 70)
    
    for addr, total in access_counts.most_common(30):
        name = addr_names.get(addr, f"tmp@{addr}")
        writes = len(writes_to_addr.get(addr, []))
        reads = len(reads_from_addr.get(addr, []))
        print(f"{addr:<10} {name:<25} {writes:<10} {reads:<10} {total:<10}")

def find_duplicate_computations(kb, ops_by_pattern, writes_to_addr):
    """Find exact duplicate computations (same op, same source addresses)"""
    
    print()
    print("=" * 70)
    print("DUPLICATE COMPUTATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Group by exact computation (op + exact source addresses)
    exact_computations = defaultdict(list)
    
    for cycle, instr in enumerate(kb.instrs):
        for engine, slots in instr.items():
            for slot in slots:
                if engine in ('valu', 'alu') and len(slot) >= 3:
                    op = slot[0]
                    dest = slot[1]
                    srcs = tuple(slot[2:])
                    key = (engine, op, srcs)
                    exact_computations[key].append((cycle, dest, slot))
    
    # Find duplicates
    duplicates = [(k, v) for k, v in exact_computations.items() if len(v) > 1]
    duplicates.sort(key=lambda x: -len(x[1]))
    
    print(f"Found {len(duplicates)} computations that appear multiple times")
    print()
    print("Top 20 duplicated computations:")
    print("-" * 70)
    
    for (engine, op, srcs), instances in duplicates[:20]:
        print(f"\n{engine}.{op}(_, {srcs}): {len(instances)} times")
        # Check if results go to different destinations
        dests = set(inst[1] for inst in instances)
        if len(dests) == 1:
            print(f"    -> Same destination each time: {list(dests)[0]}")
            print(f"    -> Could potentially be computed ONCE and reused!")
        else:
            print(f"    -> Different destinations: {sorted(dests)[:5]}{'...' if len(dests) > 5 else ''}")
        
        # Show cycles
        cycles = [inst[0] for inst in instances]
        print(f"    -> At cycles: {cycles[:10]}{'...' if len(cycles) > 10 else ''}")

if __name__ == '__main__':
    print("Analyzing kernel for value reuse opportunities...")
    print()
    
    kb, ops_by_pattern, writes_to_addr, reads_from_addr, computation_patterns = analyze_kernel()
    
    print("=" * 70)
    print("OPERATION COUNTS BY PATTERN")
    print("=" * 70)
    print()
    for pattern, count in computation_patterns.most_common(20):
        print(f"  {pattern}: {count}")
    
    find_duplicate_computations(kb, ops_by_pattern, writes_to_addr)
    find_recomputation_patterns(kb, writes_to_addr, reads_from_addr)
    find_overwritten_values(kb, writes_to_addr, reads_from_addr)
    analyze_hot_addresses(kb, writes_to_addr, reads_from_addr)
