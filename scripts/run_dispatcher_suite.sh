#!/usr/bin/env bash
# Description: Run the camera stress test with the dispatcher prototype (warm workers on every
# device, one scheduling decision per item) on the two cases of the Intel mail, with ramps past
# the 17 cameras of the national park to find the new limit:
#   A  photo cameras 4, 8, 17, 24, 32, 48, 64, 180 s measured per step;
#   B  photo + video cameras 5+5, 10+10, 20+20, 300 s measured per step.
# Configurations: round_robin and least_loaded as references, cost with the weights of the
# three use cases.
#
# Run it on the Ray head node with the cluster and the exporters (including the power
# exporter) up, in the background:
#
#   setsid bash scripts/run_dispatcher_suite.sh > /tmp/dispatcher_suite.log 2>&1 < /dev/null &
#
# Run ids: <PREFIX>_<config>_<A|B>, results in ~/wadas_results. CONFIGS and TESTS choose a subset.

set -u
cd "$(dirname "$0")/.." || exit 1

PREFIX="${PREFIX:-disp}"
CONFIGS="${CONFIGS:-roundrobin leastloaded costmon costact costcrit}"
TESTS="${TESTS:-A B}"
STEPS_A="${STEPS_A:-4+0,8+0,17+0,24+0,32+0,48+0,64+0}"
STEPS_B="${STEPS_B:-5+5,10+10,20+20}"
DETECTION_MODEL="${DETECTION_MODEL:-MDV6b-yolov9c}"
CLASSIFICATION_MODEL="${CLASSIFICATION_MODEL:-DFv1.4}"

source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors

run() {  # run <config> <use case> <policy>
  for test in $TESTS; do
    case "$test" in
      A) steps="$STEPS_A"; window=180 ;;
      B) steps="$STEPS_B"; window=300 ;;
      *) echo "Unknown test: $test"; exit 1 ;;
    esac
    echo "=== $(date +%H:%M:%S) $PREFIX $1 test $test"
    python scripts/run_camera_test.py --use-case "$2" --dispatcher "$3" \
      --steps "$steps" --window "$window" --measure-energy \
      --detection-model "$DETECTION_MODEL" --classification-model "$CLASSIFICATION_MODEL" \
      --run-id "${PREFIX}_$1_$test"
  done
}

for config in $CONFIGS; do
  case "$config" in
    roundrobin) run roundrobin MONITORING round_robin ;;
    leastloaded) run leastloaded MONITORING least_loaded ;;
    costmon) run costmon MONITORING cost ;;
    costact) run costact ACTUATOR cost ;;
    costcrit) run costcrit CRITICAL cost ;;
    *) echo "Unknown configuration: $config"; exit 1 ;;
  esac
done
echo SERIES_DONE
