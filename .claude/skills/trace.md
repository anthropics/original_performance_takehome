# /trace - Execution Trace Analysis Skill

Generate and analyze execution traces for the kernel.

## Instructions

When the user runs `/trace`, perform the following:

### 1. Generate the trace
```bash
python perf_takehome.py Tests.test_kernel_trace
```
This creates `trace.json` in the project root.

### 2. Report trace generation
- Confirm trace.json was created
- Report the cycle count from the output

### 3. Provide visualization instructions
Tell the user how to view the trace:
```
To visualize the trace:
1. In another terminal, run: python watch_trace.py
2. A browser will open - click "Open Perfetto"
3. The trace will load automatically
4. Re-run the trace command to hot-reload new traces
```

### 4. Trace Analysis Tips
Provide these analysis tips:
- Look at slot utilization per engine (alu, valu, load, store, flow)
- Identify cycles with low utilization (optimization opportunities)
- Check for sequential dependencies that block parallelism
- Look for patterns that could be vectorized

### 5. Optional: Quick trace analysis
If the user asks for analysis, read trace.json and provide:
- Total cycles
- Average slots used per cycle per engine
- Cycles where load/store slots are saturated (bottleneck indicator)
- Longest runs of single-slot instructions (packing opportunities)

## Understanding the Trace

The trace shows:
- **Rows**: Different engine slots (alu-0 through alu-11, valu-0 through valu-5, etc.)
- **Columns**: Time (cycles)
- **Blocks**: Individual operations executing in each slot

Look for:
- Empty rows = unused parallelism
- Sparse columns = low utilization cycles
- Dense columns = well-packed instructions
