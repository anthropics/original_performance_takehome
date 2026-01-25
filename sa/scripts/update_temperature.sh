#!/bin/bash
# Update temperature using cooling schedule
# Usage: sa/scripts/update_temperature.sh <current_temp> <cooling_rate>
# Output: New temperature value
#
# Uses geometric cooling: T_new = T_current * cooling_rate

if [ -z "$2" ]; then
    echo "Usage: $0 <current_temp> <cooling_rate>"
    exit 1
fi

CURRENT_TEMP="$1"
COOLING_RATE="$2"

# Calculate new temperature
python3 << EOF
current = $CURRENT_TEMP
rate = $COOLING_RATE
new_temp = current * rate
print(f"{new_temp:.6f}")
EOF
