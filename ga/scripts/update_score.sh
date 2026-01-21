#!/bin/bash
# Update a single candidate's score (GA wrapper)
# Usage: ga/scripts/update_score.sh <candidate_id> <cycles>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"

exec "$PROJECT_ROOT/scripts/update_score.sh" ga "$@"
