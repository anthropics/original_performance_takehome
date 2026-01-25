#!/usr/bin/env bash
# Calculate mutation step size category based on GA generation progress
# Usage: calc_step_size_ga.sh <current_gen> <max_gen>
#
# Uses LOGARITHMIC mapping similar to SA's temperature-based approach.
# Early generations use larger step sizes for exploration (diversity),
# later generations use smaller step sizes for exploitation (fine-tuning).
#
# Categories (higher = bolder changes allowed):
#   minimal     - single tweaks, fine-tuning
#   small       - local changes
#   moderate    - focused optimizations
#   substantial - restructuring
#   extensive   - major approach changes

set -euo pipefail

CURRENT_GEN="${1:?Usage: calc_step_size_ga.sh <current_gen> <max_gen>}"
MAX_GEN="${2:?Usage: calc_step_size_ga.sh <current_gen> <max_gen>}"

python3 -c "
current_gen = int('$CURRENT_GEN')
max_gen = int('$MAX_GEN')

# Handle edge cases
if max_gen <= 1 or current_gen <= 0:
    category = 'extensive'
elif current_gen >= max_gen:
    category = 'minimal'
else:
    # Linear progress mapping (0 at gen 1, 1 at max_gen)
    # GA generations are linear, unlike SA's exponential temperature decay
    progress = (current_gen - 1) / (max_gen - 1) if max_gen > 1 else 0

    # Balanced distribution across categories:
    # First 20%: extensive (exploration/diversity)
    # 20-40%: substantial
    # 40-60%: moderate
    # 60-80%: small
    # Last 20%: minimal (fine-tuning)
    if progress < 0.2:
        category = 'extensive'
    elif progress < 0.4:
        category = 'substantial'
    elif progress < 0.6:
        category = 'moderate'
    elif progress < 0.8:
        category = 'small'
    else:
        category = 'minimal'

print(category)
"
