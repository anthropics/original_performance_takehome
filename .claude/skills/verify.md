# /verify - Submission Verification Skill

Verify that the current implementation is valid for submission.

## Instructions

When the user runs `/verify`, perform the following checks:

### 1. Check that tests/ folder is unchanged
```bash
git diff origin/main tests/
```
This MUST produce empty output. If there are any changes to tests/, the submission is INVALID.

### 2. Run the official submission tests
```bash
python tests/submission_tests.py
```

### 3. Report Results
After running the tests, provide a summary:
- Whether tests/ folder is unchanged (CRITICAL)
- Current cycle count
- Speedup over baseline (147,734 cycles)
- Which performance thresholds are passed:
  - [ ] < 147,734 (baseline)
  - [ ] < 18,532 (updated starting point)
  - [ ] < 2,164 (Opus 4 many hours)
  - [ ] < 1,790 (Opus 4.5 casual)
  - [ ] < 1,579 (Opus 4.5 2hr)
  - [ ] < 1,548 (Sonnet 4.5 many hours)
  - [ ] < 1,487 (Opus 4.5 11hr)
  - [ ] < 1,363 (Opus 4.5 improved harness)

### Important Notes
- NEVER modify any files in tests/
- Always verify tests/ integrity before reporting results
- If correctness tests fail, the cycle count is invalid
