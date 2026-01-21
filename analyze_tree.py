"""Analyze tree structure to find patterns that might enable optimization"""

from problem import Tree, Input, build_mem_image, reference_kernel, myhash
import random

def analyze_tree_patterns(forest_height=10, batch_size=256, rounds=16, seed=123):
    """Analyze patterns in tree traversal"""
    random.seed(seed)

    # Create tree and input
    tree = Tree.generate(forest_height)
    n_nodes = 2 ** (forest_height + 1) - 1
    inp = Input(
        indices=[random.randrange(n_nodes) for _ in range(batch_size)],
        values=[random.randrange(2**32) for _ in range(batch_size)],
        rounds=rounds,
    )

    # Track all unique node accesses
    node_access_counts = {}
    node_access_by_round = [set() for _ in range(rounds)]

    # Simulate and track
    for r in range(rounds):
        for i in range(batch_size):
            idx = inp.indices[i]
            node_access_counts[idx] = node_access_counts.get(idx, 0) + 1
            node_access_by_round[r].add(idx)

            val = inp.values[i]
            val = myhash(val ^ tree.values[idx])
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            inp.values[i] = val
            inp.indices[i] = idx

    print(f"Total nodes accessed: {len(node_access_counts)} out of {n_nodes}")
    print(f"Average accesses per node: {sum(node_access_counts.values()) / len(node_access_counts):.2f}")
    print(f"Max accesses to single node: {max(node_access_counts.values())}")

    # Check if indices converge
    for r in range(rounds):
        print(f"Round {r}: {len(node_access_by_round[r])} unique nodes accessed")

    # Check depth distribution
    depth_counts = {}
    for idx in node_access_counts.keys():
        depth = idx.bit_length() - 1 if idx > 0 else 0
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print(f"\nDepth distribution of accessed nodes:")
    for depth in sorted(depth_counts.keys()):
        print(f"  Depth {depth}: {depth_counts[depth]} nodes")

    # Analyze hottest nodes
    hot_nodes = sorted(node_access_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"\nTop 20 hottest nodes:")
    total_hot_accesses = 0
    for idx, count in hot_nodes:
        total_hot_accesses += count
        print(f"  Node {idx} (bin: {bin(idx)}): {count} accesses")
    print(f"Top 20 nodes account for {total_hot_accesses}/4096 = {100*total_hot_accesses/4096:.1f}% of accesses")

    # Check if hot nodes are near root
    hot_node_depths = [idx.bit_length()-1 if idx>0 else 0 for idx, _ in hot_nodes[:20]]
    print(f"Depths of hot nodes: {sorted(set(hot_node_depths))}")

    # Check for cycles/fixed points in later rounds
    print(f"\nChecking for convergence to fixed points:")
    random.seed(seed)
    tree2 = Tree.generate(forest_height)
    inp2 = Input(
        indices=[random.randrange(n_nodes) for _ in range(batch_size)],
        values=[random.randrange(2**32) for _ in range(batch_size)],
        rounds=rounds,
    )

    # Save state at rounds 15, 16, 20, 30
    states = {}
    for r in range(40):
        if r in [15, 16, 17, 20, 30]:
            states[r] = (inp2.indices[:], inp2.values[:])

        for i in range(batch_size):
            idx = inp2.indices[i]
            val = inp2.values[i]
            val = myhash(val ^ tree2.values[idx])
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            inp2.values[i] = val
            inp2.indices[i] = idx

    # Check if states stabilize
    if 15 in states and 16 in states:
        changes_15_16 = sum(1 for i in range(batch_size) if states[15][0][i] != states[16][0][i])
        print(f"  Changes from round 15→16: {changes_15_16}/{batch_size} indices changed")
    if 16 in states and 17 in states:
        changes_16_17 = sum(1 for i in range(batch_size) if states[16][0][i] != states[17][0][i])
        print(f"  Changes from round 16→17: {changes_16_17}/{batch_size} indices changed")
    if 17 in states and 20 in states:
        changes_17_20 = sum(1 for i in range(batch_size) if states[17][0][i] != states[20][0][i])
        print(f"  Changes from round 17→20: {changes_17_20}/{batch_size} indices changed")

if __name__ == "__main__":
    analyze_tree_patterns()
