#!/usr/bin/env python3
"""
Parameter sweep to find optimal group_size and round_tile values.
"""
import sys
sys.path.insert(0, '/home/travers/brain-2/01-projects/original_performance_takehome')

from perf_takehome import KernelBuilder, do_kernel_test
import random

def test_params(group_size, round_tile):
    """Test specific parameter combination."""
    try:
        random.seed(123)
        cycles = do_kernel_test(10, 16, 256)
        return cycles
    except Exception as e:
        return None

def main():
    print("Parameter Sweep: group_size vs round_tile")
    print("=" * 70)
    
    # Current baseline
    baseline_gs, baseline_rt = 17, 13
    print(f"\nBaseline (gs={baseline_gs}, rt={baseline_rt}):")
    baseline_cycles = test_params(baseline_gs, baseline_rt)
    print(f"  → {baseline_cycles} cycles\n")
    
    results = []
    
    # Sweep promising ranges
    for gs in range(14, 21):  # group_size: 14-20
        for rt in range(10, 17):  # round_tile: 10-16
            if gs == baseline_gs and rt == baseline_rt:
                continue  # Skip baseline, already tested
            
            print(f"Testing gs={gs:2d}, rt={rt:2d}...", end=' ', flush=True)
            cycles = test_params(gs, rt)
            
            if cycles is None:
                print("FAILED")
            else:
                delta = cycles - baseline_cycles
                symbol = "✓" if delta < 0 else "✗" if delta > 0 else "="
                print(f"{cycles:4d} cycles ({delta:+4d}) {symbol}")
                results.append((cycles, gs, rt, delta))
    
    # Summary
    print("\n" + "=" * 70)
    print("TOP 10 RESULTS:")
    print("=" * 70)
    results.sort()
    for i, (cycles, gs, rt, delta) in enumerate(results[:10], 1):
        star = " ★" if i == 1 else ""
        print(f"{i:2d}. gs={gs:2d}, rt={rt:2d} → {cycles:4d} cycles ({delta:+4d}){star}")
    
    if results and results[0][0] < baseline_cycles:
        print(f"\n🎉 Found improvement: gs={results[0][1]}, rt={results[0][2]} → {results[0][0]} cycles!")
    else:
        print(f"\n✓ Baseline (gs={baseline_gs}, rt={baseline_rt}) is optimal")

if __name__ == "__main__":
    main()
