#!/usr/bin/env python3
"""Validate speculative round computation logic."""

import random
import sys

sys.path.insert(0, "/home/travers/brain-2/01-projects/original_performance_takehome")

from problem import Tree, Input, HASH_STAGES, myhash


def test_speculative_selection():
    """Test that arithmetic selection gives correct node values for rounds 1, 12."""
    random.seed(123)
    forest = Tree.generate(10)

    tree1 = forest.values[1]
    tree2 = forest.values[2]
    delta = (tree2 - tree1) % (2**32)

    print(f"tree[1] = {tree1}")
    print(f"tree[2] = {tree2}")
    print(f"delta = tree[2] - tree[1] = {delta}")
    print()

    # Simulate round 0 to get indices for round 1
    inp = Input.generate(forest, 16, 16)  # Small batch for testing

    # After round 0, all indices should be 1 or 2
    indices_after_r0 = []
    for i in range(len(inp.indices)):
        idx = inp.indices[i]  # 0
        val = inp.values[i]
        val = myhash(val ^ forest.values[idx])
        new_idx = 2 * idx + (1 if val % 2 == 0 else 2)
        new_idx = 0 if new_idx >= len(forest.values) else new_idx
        indices_after_r0.append(new_idx)

    print(f"Indices after round 0: {indices_after_r0}")
    print(f"All in {{1, 2}}: {all(i in {1, 2} for i in indices_after_r0)}")
    print()

    # Test arithmetic selection
    print("Testing arithmetic selection:")
    for idx in [1, 2]:
        selector = idx - 1  # 0 if idx==1, 1 if idx==2
        node_arith = (tree1 + delta * selector) % (2**32)
        node_actual = forest.values[idx]
        match = "✓" if node_arith == node_actual else "✗"
        print(
            f"  idx={idx}: selector={selector}, arith={node_arith}, actual={node_actual} {match}"
        )


def test_full_round_comparison():
    """Compare speculative vs normal round processing."""
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 8, 16)

    # Process round 0 normally
    vals = inp.values.copy()
    idxs = inp.indices.copy()

    for i in range(len(idxs)):
        idx = idxs[i]
        val = vals[i]
        val = myhash(val ^ forest.values[idx])
        idx = 2 * idx + (1 if val % 2 == 0 else 2)
        idx = 0 if idx >= len(forest.values) else idx
        vals[i] = val
        idxs[i] = idx

    print(f"After round 0:")
    print(f"  indices: {idxs}")
    print(f"  values: {vals}")
    print()

    # Process round 1 - NORMAL way (gather)
    vals_normal = vals.copy()
    idxs_normal = idxs.copy()

    for i in range(len(idxs_normal)):
        idx = idxs_normal[i]
        val = vals_normal[i]
        node_val = forest.values[idx]
        val = myhash(val ^ node_val)
        idx = 2 * idx + (1 if val % 2 == 0 else 2)
        idx = 0 if idx >= len(forest.values) else idx
        vals_normal[i] = val
        idxs_normal[i] = idx

    print(f"After round 1 (NORMAL):")
    print(f"  indices: {idxs_normal}")
    print(f"  values: {vals_normal}")
    print()

    # Process round 1 - SPECULATIVE way (arithmetic selection)
    vals_spec = vals.copy()
    idxs_spec = idxs.copy()

    tree1 = forest.values[1]
    tree2 = forest.values[2]
    delta = (tree2 - tree1) % (2**32)

    for i in range(len(idxs_spec)):
        idx = idxs_spec[i]
        val = vals_spec[i]

        # Arithmetic selection: node = tree1 + delta * (idx - 1)
        selector = idx - 1
        node_val = (tree1 + delta * selector) % (2**32)

        val = myhash(val ^ node_val)
        idx = 2 * idx + (1 if val % 2 == 0 else 2)
        idx = 0 if idx >= len(forest.values) else idx
        vals_spec[i] = val
        idxs_spec[i] = idx

    print(f"After round 1 (SPECULATIVE):")
    print(f"  indices: {idxs_spec}")
    print(f"  values: {vals_spec}")
    print()

    # Compare
    match = vals_normal == vals_spec and idxs_normal == idxs_spec
    print(f"Results match: {match}")

    if not match:
        print("MISMATCH DETAILS:")
        for i in range(len(vals_normal)):
            if vals_normal[i] != vals_spec[i] or idxs_normal[i] != idxs_spec[i]:
                print(
                    f"  i={i}: normal=({idxs_normal[i]}, {vals_normal[i]}), spec=({idxs_spec[i]}, {vals_spec[i]})"
                )


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Speculative Selection Arithmetic")
    print("=" * 60)
    test_speculative_selection()
    print()
    print("=" * 60)
    print("TEST 2: Full Round Comparison")
    print("=" * 60)
    test_full_round_comparison()
