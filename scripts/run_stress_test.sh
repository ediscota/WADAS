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
#
# CONFIGS chooses the configurations (default: the four above). Two more study the NPU:
#   npupin     detection pinned on the NPU of wadas1, classification on the GPU of wadas2: how
#              many continuous streams one NPU sustains
#   monshared  Resource-Aware on the energy profile with the NPU shared between detections
#              (npu_shared), against monenergy where an NPU takes one task at a time

set -u
cd "$(dirname "$0")/.." || exit 1

PREFIX="${PREFIX:-st}"
REPS="${REPS:-1}"
STEPS="${STEPS:-1,2,4,6,8,12,16,20,24,32}"
CONFIGS="${CONFIGS:-monorig monenergy actpend critenergy}"
DETECTION_MODEL="${DETECTION_MODEL:-MDV6b-yolov9c}"
CLASSIFICATION_MODEL="${CLASSIFICATION_MODEL:-DFv1.4}"
PROFILE="${PROFILE:-scripts/energy_profile_mdv6b_dfv14.json}"

source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors

ENERGY="{\"energy_profile\": \"$PROFILE\", \"pending_penalty\": 5}"
SHARED="{\"energy_profile\": \"$PROFILE\", \"pending_penalty\": 5, \"npu_shared\": true}"

run() {  # run <config> <rep> <use case> <scheduler params>
  echo "=== $(date +%H:%M:%S) $PREFIX $1 rep $2"
  python scripts/run_stress.py --use-case "$3" --scheduler-params "$4" --steps "$STEPS" \
    --measure-energy --detection-model "$DETECTION_MODEL" \
    --classification-model "$CLASSIFICATION_MODEL" --run-id "${PREFIX}_$1_$2"
}

for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then
    configs="$CONFIGS"
  else
    configs=""
    for c in $CONFIGS; do configs="$c $configs"; done
  fi
  for config in $configs; do
    case "$config" in
      monorig) run monorig "$rep" MONITORING '{}' ;;
      monenergy) run monenergy "$rep" MONITORING "$ENERGY" ;;
      actpend) run actpend "$rep" ACTUATOR '{"pending_penalty": 0.05}' ;;
      critenergy) run critenergy "$rep" CRITICAL "$ENERGY" ;;
      npupin) run npupin "$rep" PINNED '{"detection": "wadas1:NPU", "classification": "wadas2:GPU"}' ;;
      monshared) run monshared "$rep" MONITORING "$SHARED" ;;
      *) echo "Unknown configuration: $config"; exit 1 ;;
    esac
  done
done
echo SERIES_DONE
