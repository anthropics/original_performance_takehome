#!/bin/bash
# Copy a candidate folder to a new ID (SA wrapper)
# Usage: sa/scripts/copy_candidate.sh <source_id> <dest_id>
# Output: "Copied {source} to {dest}"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SA_DIR")"

exec "$PROJECT_ROOT/scripts/copy_candidate.sh" sa "$@"
