#!/bin/bash
# Initialize a candidate folder (SA wrapper)
# Usage: sa/scripts/init_candidate.sh <ID>

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SA_DIR")"

exec "$PROJECT_ROOT/scripts/init_candidate.sh" sa "$@"
