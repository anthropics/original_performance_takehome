#!/bin/bash
# Get the next available candidate ID (SA wrapper)
# Usage: sa/scripts/next_candidate_id.sh
# Output: Next available ID (e.g., "011" if 001-010 exist)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SA_DIR")"

exec "$PROJECT_ROOT/scripts/next_candidate_id.sh" sa
