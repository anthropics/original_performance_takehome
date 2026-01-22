# /optimize - Optimization Cycle Skill

Run a structured optimization cycle to reduce cycle count.

## Instructions

When the user runs `/optimize`, follow this structured approach:

### Phase 1: Baseline Assessment
1. Run `/benchmark` to get current cycle count
2. Read the current `perf_takehome.py` implementation
3. Identify the current optimization level:
   - Scalar only, no packing (baseline)
   - Scalar with instruction packing
   - Vectorized (VLEN=8)
   - Vectorized with packing
   - Fully optimized

### Phase 2: Identify Optimization Opportunities
Analyze the code for:

**Low-hanging fruit:**
- [ ] Instruction packing (multiple slots per cycle)
- [ ] Vectorization (VLEN=8 elements at once)
- [ ] Constant hoisting (compute once, reuse)
- [ ] Loop unrolling

**Medium complexity:**
- [ ] Memory access optimization (minimize loads/stores)
- [ ] Pipeline optimization (overlap operations)
- [ ] Scratch space management
- [ ] Dependency chain reduction

**Advanced:**
- [ ] Software pipelining across iterations
- [ ] Custom instruction sequences
- [ ] Algorithm-level optimizations

### Phase 3: Implement ONE Optimization
Pick the highest-impact optimization that isn't yet implemented and apply it:
1. Make the code change to `perf_takehome.py`
2. Run `/verify` to check correctness AND performance
3. Report the improvement (or rollback if broken)

### Phase 4: Report
Provide a summary:
- Previous cycle count
- New cycle count
- Improvement (cycles saved, percentage)
- What optimization was applied
- Suggested next optimization

## Optimization Priority Order

Typical order of impact (highest first):
1. **Vectorization**: 8x theoretical improvement
2. **Instruction packing**: Up to slot_limit x improvement
3. **Loop optimization**: Reduces overhead
4. **Memory optimization**: Load/store are bottlenecks (only 2 slots each)
5. **Fine-tuning**: Dependency chains, scheduling

## Safety Rules

1. **NEVER modify tests/** - Check with `git diff origin/main tests/`
2. **Always verify correctness** - Run submission_tests.py
3. **Make incremental changes** - One optimization at a time
4. **Keep backups** - Use git commits between optimizations

## Architecture Quick Reference

```
Slot Limits: alu=12, valu=6, load=2, store=2, flow=1
Vector Length: VLEN=8
Scratch Space: 1536 words
Single Core Only: N_CORES=1

Memory Layout:
  mem[4] = forest_values_p (tree node values)
  mem[5] = inp_indices_p (current indices, 256 items)
  mem[6] = inp_values_p (current values, 256 items)

Test Parameters:
  forest_height=10 (2047 nodes)
  rounds=16
  batch_size=256
```
