#!/bin/bash
# Initialize a candidate folder (GA wrapper)
# Usage: ga/scripts/init_candidate.sh <ID>

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GA_DIR")"

exec "$PROJECT_ROOT/scripts/init_candidate.sh" ga "$@"
