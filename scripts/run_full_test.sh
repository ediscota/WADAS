#!/usr/bin/env bash
# Description: Run the complete scheduler comparison on the Ray cluster: three loads (photos,
# videos, mix), the three schedulers without and with the recent-placement penalty, and the
# energy-aware Resource-Aware scheduler, with the energy of every scenario measured.
#
# Run it on the Ray head node with the cluster and the exporters (including the power
# exporter) up, in the background so that it survives the end of the ssh session:
#
#   setsid bash scripts/run_full_test.sh > /tmp/full_test.log 2>&1 < /dev/null &
#
# Every setting runs in two passes of REPS_PER_PASS repetitions, passes in a different order,
# so that a slow drift of the machines (idle power, temperature) does not favour one setting.
# The run ids are  <PREFIX>_<load>_<variant>_<pass>  and the results end up in
# ~/wadas_results. The cluster must have 5 nodes before every run, or the test stops.

set -u
cd "$(dirname "$0")/.." || exit 1

PREFIX="${PREFIX:-mt}"
REPS_PER_PASS="${REPS_PER_PASS:-5}"
DETECTION_MODEL="${DETECTION_MODEL:-MDV6b-yolov9c}"
CLASSIFICATION_MODEL="${CLASSIFICATION_MODEL:-DFv1.4}"
ENERGY_PROFILE="${ENERGY_PROFILE:-scripts/energy_profile_mdv6b_dfv14.json}"
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

run() {  # run <load> <variant> <pass> <seed> <use cases> [scheduler params]
  local load="$1" variant="$2" pass="$3" seed="$4" cases="$5" params="${6:-{\}}"
  check_cluster
  echo "=== $(date +%H:%M:%S) $PREFIX $load $variant pass $pass"
  # shellcheck disable=SC2046
  python scripts/run_scenario.py "${COMMON[@]}" $(load_args "$load") \
    --repeats "$REPS_PER_PASS" --seed "$seed" --use-cases $cases \
    --scheduler-params "$params" --run-id "${PREFIX}_${load}_${variant}_${pass}"
}

PENALTY_PROFILE="\"energy_profile\": \"$ENERGY_PROFILE\""

for pass in a b; do
  if [ "$pass" = a ]; then loads="photos mix videos"; seed=0; else loads="videos mix photos"; seed=1; fi
  for load in $loads; do
    run "$load" base "$pass" "$seed" "MONITORING ACTUATOR CRITICAL"
    run "$load" pend "$pass" "$seed" "MONITORING CRITICAL" '{"pending_penalty": 10}'
    run "$load" pendact "$pass" "$seed" "ACTUATOR" '{"pending_penalty": 0.05}'
    run "$load" ea3 "$pass" "$seed" "MONITORING" "{$PENALTY_PROFILE, \"pending_penalty\": 3}"
    run "$load" ea5 "$pass" "$seed" "MONITORING" "{$PENALTY_PROFILE, \"pending_penalty\": 5}"
  done
done
echo SERIES_DONE
