# Anthropic's Original Performance Take-Home

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

The original take-home was a 4-hour one that starts close to the contents of this repo, after Claude Opus 4 beat most humans at that, it was updated to a 2-hour one which started with code which achieved 18532 cycles (7.97x faster than this repo starts you). This repo is based on the newer take-home which has a few more instructions and comes with better debugging tools, but has the starter code reverted to the slowest baseline. After Claude Opus 4.5 we started using a different base for our time-limited take-homes.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks 

Measured in clock cycles from the simulated machine. All of these numbers are for models doing the 2 hour version which started at 18532 cycles:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness
- **??? cycles**: Best human performance ever is substantially better than the above, but we won't say how much.

While it's no longer a good time-limited test, you can still use this test to get us excited about hiring you! If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed, especially if you get near the best solution we've seen. New model releases may change what threshold impresses us though, and no guarantees that we keep this readme updated with the latest on that.

Run `python tests/submission_tests.py` to see which thresholds you pass.

## Leaderboard submission

This branch is set up for copy-paste submission:

- `perf_takehome.py` contains the validated submission wired into the repo.
- `build_kernel_submission.txt` contains the exact self-contained `build_kernel` method to paste into the leaderboard editor.

Validation on this branch:

```bash
python3 tests/submission_tests.py
```

The current frozen submission result is **1,319 cycles** for the standard `forest_height=10`, `rounds=16`, `batch_size=256` benchmark. The same copy-paste method was also checked for correctness across tree depths 8-10, rounds 8/12/16/20, and batch sizes 128/256.

### Techniques used

The submission is a self-contained method: it defines the optimized helper builder inside `build_kernel`, generates the VLIW program, then copies the generated instructions and scratch metadata back onto the starter `KernelBuilder`.

The main performance techniques are:

- Keep all batch values and indices resident in scratch across rounds, then store final values once.
- Vectorize the batch in `VLEN=8` chunks and schedule across multiple chunk groups.
- Use a critical-path list scheduler with explicit scratch and memory dependency tracking.
- Specialize cheap forest levels: root broadcast, level-1 select, level-2 pair select, and partial level-3 select.
- Use an `8x3` temporary-bank layout to keep enough independent chunks in flight without exceeding the 1,536-word scratch limit.
- Balance hash work between ALU and VALU with per-level and per-round masks.
- Move one level-0 branch-update chunk onto ALU to fill scheduler slack.

### Profiling lessons

The most useful profiling result was separating **resource floors** from **scheduled cycles**. The optimized kernel has a much lower theoretical floor than the emitted schedule, but after the major structural wins most local scheduler and mask tweaks produced only single-digit gains.

Specific lessons:

- Forest gathers dominate early until resident state and level-specialized loads remove avoidable traffic.
- Generic forest caches and broad compare/select distribution are usually too expensive; saving load words is not enough if VALU/flow distribution costs more.
- Scratch pressure is a first-order constraint. Many promising ideas failed because the final useful shape already uses about 1,534 / 1,536 scratch words.
- Scheduler changes should be guided by DAG/resource stats, not by broad priority guesses. Tie-break and weighted-priority variants mostly tied or regressed.
- Once the load/VALU/ALU floors are close, per-round masks can matter more than per-level defaults because late-round slack differs from early-round slack.

## Warning: LLMs can cheat

None of the solutions we received on the first day post-release below 1300 cycles were valid solutions. In each case, a language model modified the tests to make the problem easier.

If you use an AI agent, we recommend instructing it not to change the `tests/` folder and to use `tests/submission_tests.py` for verification.

Please run the following commands to validate your submission, and mention that you did so when submitting:
```
# This should be empty, the tests folder must be unchanged
git diff origin/main tests/
# You should pass some of these tests and use the cycle count this prints
python tests/submission_tests.py
```

An example of this kind of hack is a model noticing that `problem.py` has multicore support, implementing multicore as an optimization, noticing there's no speedup and "debugging" that `N_CORES = 1` and "fixing" the core count so they get a speedup. Multicore is disabled intentionally in this version.
