#!/usr/bin/env bash
# Description: Measure the steady-state profile of every device: time and energy per processed
# item with the model already loaded, as a warm worker of the dispatcher prototype would see it.
#
# Each run pins detection on one device and classification on a GPU of another node (so that the
# energy of each node belongs to one role) and sends one stream at FPS frames per second, for
# every FPS of RATES: at the highest rate the devices are saturated. Then:
#
#   python scripts/build_steady_profile.py ~/wadas_results/sp_* --out scripts/steady_profile.json
#
# Run it on the Ray head node with the cluster and the exporters (including the power
# exporter) up, in the background:
#
#   setsid bash scripts/run_steady_profile.sh > /tmp/steady_profile.log 2>&1 < /dev/null &
#
# Run ids: <PREFIX>_<detection>_<classification>_f<fps>, results in ~/wadas_results.

set -u
cd "$(dirname "$0")/.." || exit 1

PREFIX="${PREFIX:-sp}"
RATES="${RATES:-2 4 8}"
WINDOW="${WINDOW:-120}"
DETECTION_MODEL="${DETECTION_MODEL:-MDV6b-yolov9c}"
CLASSIFICATION_MODEL="${CLASSIFICATION_MODEL:-DFv1.4}"
# detection device, classification device: every detection device once, every GPU at least
# once for classification, the two always on different nodes
PAIRS="${PAIRS:-wadas1:NPU,wadas3:GPU wadas2:NPU,wadas4:GPU wadas1:GPU,wadas5:GPU \
wadas2:GPU,wadas1:GPU wadas3:GPU,wadas2:GPU wadas4:GPU,wadas3:GPU wadas5:GPU,wadas4:GPU}"

source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors

for fps in $RATES; do
  for pair in $PAIRS; do
    det="${pair%,*}"
    cls="${pair#*,}"
    name="${PREFIX}_${det/:/-}_${cls/:/-}_f${fps}"
    echo "=== $(date +%H:%M:%S) $name"
    python scripts/run_stress.py --use-case PINNED \
      --scheduler-params "{\"detection\": \"$det\", \"classification\": \"$cls\"}" \
      --steps 1 --fps "$fps" --window "$WINDOW" --extra-steps 0 --measure-energy \
      --detection-model "$DETECTION_MODEL" --classification-model "$CLASSIFICATION_MODEL" \
      --run-id "$name"
  done
done
echo SERIES_DONE
