#!/bin/bash
# Update a single candidate's score (SA wrapper)
# Usage: sa/scripts/update_score.sh <candidate_id> <cycles>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SA_DIR")"

exec "$PROJECT_ROOT/scripts/update_score.sh" sa "$@"
