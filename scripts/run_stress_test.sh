#!/usr/bin/env bash
# Description: Run the continuous-stream stress test (scripts/run_stress.py) on the four
# scheduler configurations presented to Intel: the original Resource-Aware as a reference and
# the three proposed ones. Each configuration runs a ramp of streams at 1 frame/s until it
# stops being sustainable.
#
# Run it on the Ray head node with the cluster and the exporters (including the power
# exporter) up, in the background so that it survives the end of the ssh session:
#
#   setsid bash scripts/run_stress_test.sh > /tmp/stress_test.log 2>&1 < /dev/null &
#
# Run ids: <PREFIX>_<config>_<rep>, results in ~/wadas_results. Repetitions alternate the
# order of the configurations, so that a slow drift of the machines does not favour one.

set -u
cd "$(dirname "$0")/.." || exit 1

PREFIX="${PREFIX:-st}"
REPS="${REPS:-1}"
STEPS="${STEPS:-1,2,4,6,8,12,16,20,24,32}"
DETECTION_MODEL="${DETECTION_MODEL:-MDV6b-yolov9c}"
CLASSIFICATION_MODEL="${CLASSIFICATION_MODEL:-DFv1.4}"
PROFILE="${PROFILE:-scripts/energy_profile_mdv6b_dfv14.json}"

source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors

ENERGY="{\"energy_profile\": \"$PROFILE\", \"pending_penalty\": 5}"

run() {  # run <config> <rep> <use case> <scheduler params>
  echo "=== $(date +%H:%M:%S) $PREFIX $1 rep $2"
  python scripts/run_stress.py --use-case "$3" --scheduler-params "$4" --steps "$STEPS" \
    --measure-energy --detection-model "$DETECTION_MODEL" \
    --classification-model "$CLASSIFICATION_MODEL" --run-id "${PREFIX}_$1_$2"
}

for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then
    configs="monorig monenergy actpend critenergy"
  else
    configs="critenergy actpend monenergy monorig"
  fi
  for config in $configs; do
    case "$config" in
      monorig) run monorig "$rep" MONITORING '{}' ;;
      monenergy) run monenergy "$rep" MONITORING "$ENERGY" ;;
      actpend) run actpend "$rep" ACTUATOR '{"pending_penalty": 0.05}' ;;
      critenergy) run critenergy "$rep" CRITICAL "$ENERGY" ;;
    esac
  done
done
echo SERIES_DONE
