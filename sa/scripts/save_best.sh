#!/bin/bash
# Save a candidate as the current best (SA wrapper)
# Usage: sa/scripts/save_best.sh <candidate_id>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SA_DIR")"

exec "$PROJECT_ROOT/scripts/save_best.sh" sa "$@"
