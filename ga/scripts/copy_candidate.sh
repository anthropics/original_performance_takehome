#!/bin/bash
# Copy a candidate folder to a new ID (GA wrapper)
# Usage: ga/scripts/copy_candidate.sh <source_id> <dest_id>
# Output: "Copied {source} to {dest}"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"

exec "$PROJECT_ROOT/scripts/copy_candidate.sh" ga "$@"
