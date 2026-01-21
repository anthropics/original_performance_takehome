#!/bin/bash
# Evaluate a candidate and return cycle count
# Usage: scripts/eval_candidate.sh <candidate_id>
# Output: cycle count (integer) or "ERROR" on failure
#
# Optimization: Caches results by file hash to skip duplicate evaluations

if [ -z "$1" ]; then
    echo "Usage: $0 <candidate_id>"
    exit 1
fi

ID="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CANDIDATE_DIR="$ROOT_DIR/candidates/CAND_$ID"
CACHE_FILE="$ROOT_DIR/candidates/eval_cache.txt"

if [ ! -d "$CANDIDATE_DIR" ]; then
    echo "ERROR: Candidate $ID does not exist"
    exit 1
fi

# Compute hash of the kernel code (only build_kernel matters)
CODE_FILE="$CANDIDATE_DIR/perf_takehome.py"
if [ ! -f "$CODE_FILE" ]; then
    echo "ERROR: perf_takehome.py not found"
    exit 1
fi

# Use md5sum for fast hashing
HASH=$(md5sum "$CODE_FILE" | cut -d' ' -f1)

# Check cache
touch "$CACHE_FILE"  # Ensure file exists
CACHED=$(grep "^$HASH " "$CACHE_FILE" | head -1 | cut -d' ' -f2)

if [ -n "$CACHED" ]; then
    # Cache hit - return cached result
    echo "$CACHED"
    exit 0
fi

# Cache miss - run evaluation
cd "$CANDIDATE_DIR"
RESULT=$(python submission_tests.py 2>&1 || true)

# Extract cycle count from output (first CYCLES line)
CYCLES=$(echo "$RESULT" | grep "CYCLES:" | head -1 | awk '{print $2}')

if [ -z "$CYCLES" ]; then
    echo "ERROR"
    exit 1
fi

# Store in cache
echo "$HASH $CYCLES" >> "$CACHE_FILE"

echo "$CYCLES"
