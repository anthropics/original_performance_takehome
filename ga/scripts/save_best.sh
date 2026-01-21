#!/bin/bash
# Save a candidate as the current best (GA wrapper)
# Usage: ga/scripts/save_best.sh <candidate_id>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"

exec "$PROJECT_ROOT/scripts/save_best.sh" ga "$@"
