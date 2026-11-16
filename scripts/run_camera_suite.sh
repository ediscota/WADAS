#!/usr/bin/env bash
# Description: Run the camera stress test (scripts/run_camera_test.py) on the six scheduler
# configurations, original and proposed for each use case, on the two cases of the Intel mail:
#   A  17 cameras sending photos (the national park), as a ramp 1, 2, 4, 8, 12 cameras, then 17
#      three times, 180 s measured per step;
#   B  5 cameras sending photos and 5 sending video clips into the same queue, three times,
#      300 s measured per step.
# WADAS works as in the main branch: one queue, one pipeline placed once by the scheduler.
#
# Run it on the Ray head node with the cluster and the exporters (including the power
# exporter) up, in the background so that it survives the end of the ssh session:
#
#   setsid bash scripts/run_camera_suite.sh > /tmp/camera_suite.log 2>&1 < /dev/null &
#
# Run ids: <PREFIX>_<config>_<A|B>, results in ~/wadas_results. CONFIGS and TESTS choose a subset.

set -u
cd "$(dirname "$0")/.." || exit 1

PREFIX="${PREFIX:-cam}"
CONFIGS="${CONFIGS:-monorig monenergy actorig actpend critorig critenergy}"
TESTS="${TESTS:-A B}"
STEPS_A="${STEPS_A:-1+0,2+0,4+0,8+0,12+0,17+0,17+0,17+0}"
STEPS_B="${STEPS_B:-5+5,5+5,5+5}"
DETECTION_MODEL="${DETECTION_MODEL:-MDV6b-yolov9c}"
CLASSIFICATION_MODEL="${CLASSIFICATION_MODEL:-DFv1.4}"
PROFILE="${PROFILE:-scripts/energy_profile_mdv6b_dfv14.json}"

source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors

ENERGY="{\"energy_profile\": \"$PROFILE\", \"pending_penalty\": 5}"

run() {  # run <config> <use case> <scheduler params>
  for test in $TESTS; do
    case "$test" in
      A) steps="$STEPS_A"; window=180 ;;
      B) steps="$STEPS_B"; window=300 ;;
      *) echo "Unknown test: $test"; exit 1 ;;
    esac
    echo "=== $(date +%H:%M:%S) $PREFIX $1 test $test"
    python scripts/run_camera_test.py --use-case "$2" --scheduler-params "$3" \
      --steps "$steps" --window "$window" --measure-energy \
      --detection-model "$DETECTION_MODEL" --classification-model "$CLASSIFICATION_MODEL" \
      --run-id "${PREFIX}_$1_$test"
  done
}

for config in $CONFIGS; do
  case "$config" in
    monorig) run monorig MONITORING '{}' ;;
    monenergy) run monenergy MONITORING "$ENERGY" ;;
    actorig) run actorig ACTUATOR '{}' ;;
    actpend) run actpend ACTUATOR '{"pending_penalty": 0.05}' ;;
    critorig) run critorig CRITICAL '{}' ;;
    critenergy) run critenergy CRITICAL "$ENERGY" ;;
    *) echo "Unknown configuration: $config"; exit 1 ;;
  esac
done
echo SERIES_DONE
