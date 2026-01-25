#!/bin/bash
# Evaluate a candidate (SA wrapper)
# Usage: sa/scripts/eval_candidate.sh <candidate_id>
# Output: cycle count (integer) or "ERROR" on failure

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SA_DIR")"

exec "$PROJECT_ROOT/scripts/eval_candidate.sh" sa "$@"
