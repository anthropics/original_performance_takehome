#!/bin/bash
# Evaluate a candidate (GA wrapper)
# Usage: ga/scripts/eval_candidate.sh <candidate_id>
# Output: cycle count (integer) or "ERROR" on failure

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"

exec "$PROJECT_ROOT/scripts/eval_candidate.sh" ga "$@"
