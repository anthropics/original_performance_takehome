#!/usr/bin/env bash
# Calculate mutation step size category based on temperature
# Usage: calc_step_size.sh <current_temp> <initial_temp> <final_temp>
#
# Uses LOGARITHMIC mapping to match exponential cooling schedule.
# This creates a balanced distribution across categories instead of
# spending most time in "minimal".
#
# Categories (higher = bolder changes allowed):
#   minimal     - single tweaks, fine-tuning
#   small       - local changes
#   moderate    - focused optimizations
#   substantial - restructuring
#   extensive   - major approach changes

set -euo pipefail

CURRENT_TEMP="${1:?Usage: calc_step_size.sh <current_temp> <initial_temp> <final_temp>}"
INITIAL_TEMP="${2:?Usage: calc_step_size.sh <current_temp> <initial_temp> <final_temp>}"
FINAL_TEMP="${3:?Usage: calc_step_size.sh <current_temp> <initial_temp> <final_temp>}"

python3 -c "
import math

current = float('$CURRENT_TEMP')
initial = float('$INITIAL_TEMP')
final = float('$FINAL_TEMP')

# Handle edge cases
if initial <= final or current <= final:
    category = 'minimal'
else:
    # Logarithmic mapping: matches exponential temperature decay
    # This spreads categories evenly across the cooling schedule
    log_ratio = math.log(current / final) / math.log(initial / final)
    scale = max(1, min(10, round(1 + 9 * log_ratio)))

    # Adjusted thresholds for balanced distribution:
    # ~20% extensive, ~13% substantial, ~27% moderate, ~27% small, ~12% minimal
    if scale <= 3:
        category = 'minimal'
    elif scale <= 5:
        category = 'small'
    elif scale <= 7:
        category = 'moderate'
    elif scale <= 8:
        category = 'substantial'
    else:
        category = 'extensive'

print(category)
"
