# /benchmark - Quick Performance Benchmark Skill

Run a quick performance benchmark to check current cycle count.

## Instructions

When the user runs `/benchmark`, execute the following:

### 1. Run the performance test
```bash
python perf_takehome.py Tests.test_kernel_cycles
```

### 2. Extract and Report
Parse the output to find:
- **CYCLES**: The current cycle count
- **Speedup over baseline**: How many times faster than 147,734

### 3. Quick Summary
Provide a one-line summary like:
```
Current: X,XXX cycles (YY.Xx speedup over baseline)
```

### 4. Performance Tier
Indicate which tier the current implementation falls into:
- Baseline: 147,734
- Previous starting point: 18,532
- Opus 4 many hours: 2,164
- Opus 4.5 casual: 1,790
- Opus 4.5 2hr: 1,579
- Sonnet 4.5 many hours: 1,548
- Opus 4.5 11hr: 1,487
- Best known: 1,363

### Note
This is a quick benchmark using the local simulator. For official verification, use `/verify` which runs against the frozen simulator.
