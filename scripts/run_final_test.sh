#!/usr/bin/env bash
# Description: Final comparison of the Performance-Aware (ACTUATOR) and Trade-Off (CRITICAL)
# variants on the Ray cluster, on the three loads (photos, videos, mix), all measured in the
# same series so that they can be compared directly:
#   actgpu   ACTUATOR, recent-placement penalty, original device order (GPU before NPU, the default)
#   actnpu   ACTUATOR, recent-placement penalty, NPU before GPU for detection (npu_first)
#   critpend CRITICAL, recent-placement penalty (performance per watt)
#   crite    CRITICAL on the measured energy alone (profile), penalty in joules
#   crit_et  CRITICAL on energy x time (energy and latency profiles), penalty in seconds
#
# Run it on the Ray head node with the cluster and the exporters up, in the background:
#
#   setsid bash scripts/run_final_test.sh > /tmp/final_test.log 2>&1 < /dev/null &
#
# Every setting runs in two passes of REPS_PER_PASS repetitions, passes in a different order.
# Run ids: <PREFIX>_<load>_<variant>_<pass>, results in ~/wadas_results. The cluster must
# have 5 nodes before every run, or the test stops.

set -u
cd "$(dirname "$0")/.." || exit 1

PREFIX="${PREFIX:-ft}"
REPS_PER_PASS="${REPS_PER_PASS:-5}"
DETECTION_MODEL="${DETECTION_MODEL:-MDV6b-yolov9c}"
CLASSIFICATION_MODEL="${CLASSIFICATION_MODEL:-DFv1.4}"
PROFILE="${PROFILE:-scripts/energy_profile_mdv6b_dfv14.json}"
EXPECTED_NODES="${EXPECTED_NODES:-5}"

source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors

COMMON=(--stagger 0 --cooldown 10 --measure-energy --idle-seconds 30
        --detection-model "$DETECTION_MODEL" --classification-model "$CLASSIFICATION_MODEL")

load_args() {
  case "$1" in
    photos) echo "--n-videos 0 --n-photos 10" ;;
    videos) echo "--n-videos 10" ;;
    mix) echo "--n-videos 10 --n-photos 10" ;;
  esac
}

check_cluster() {
  local nodes
  nodes=$(ray status 2>/dev/null | grep -c node_)
  if [ "$nodes" != "$EXPECTED_NODES" ]; then
    echo "ABORT: $nodes Ray nodes instead of $EXPECTED_NODES"
    exit 1
  fi
}

run() {  # run <load> <variant> <pass> <seed> <use case> <scheduler params>
  local load="$1" variant="$2" pass="$3" seed="$4" cases="$5" params="$6"
  check_cluster
  echo "=== $(date +%H:%M:%S) $PREFIX $load $variant pass $pass"
  # shellcheck disable=SC2046
  python scripts/run_scenario.py "${COMMON[@]}" $(load_args "$load") \
    --repeats "$REPS_PER_PASS" --seed "$seed" --use-cases $cases \
    --scheduler-params "$params" --run-id "${PREFIX}_${load}_${variant}_${pass}"
}

ENERGY="\"energy_profile\": \"$PROFILE\""
LATENCY="\"latency_profile\": \"$PROFILE\""

for pass in a b; do
  if [ "$pass" = a ]; then loads="mix photos videos"; seed=0; else loads="videos photos mix"; seed=1; fi
  for load in $loads; do
    run "$load" actgpu "$pass" "$seed" ACTUATOR '{"pending_penalty": 0.05}'
    run "$load" actnpu "$pass" "$seed" ACTUATOR '{"pending_penalty": 0.05, "npu_first": true}'
    run "$load" critpend "$pass" "$seed" CRITICAL '{"pending_penalty": 10}'
    run "$load" crite "$pass" "$seed" CRITICAL "{$ENERGY, \"pending_penalty\": 5}"
    run "$load" crit_et "$pass" "$seed" CRITICAL "{$ENERGY, $LATENCY, \"pending_penalty\": 0.5}"
  done
done
echo SERIES_DONE
