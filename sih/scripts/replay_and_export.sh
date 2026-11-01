#!/usr/bin/env bash
# Replay SIH CSV logs with Prototype 3D viewer and export PX4 ULog files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PROTOTYPE_ROOT="${PROTOTYPE_ROOT:-$ROOT/sim_source}"
PY_PROTO="${PROTOTYPE_ROOT}/.venv/bin/python"
if [[ ! -x "$PY_PROTO" ]]; then
  PY_PROTO=python3
fi
PY_SYS=python3

LOG_DIR="$ROOT/sih/logs"
VEHICLE="${1:-$ROOT/config/vehicles/tvc_ring3_g12.json}"
if [[ ! -f "$VEHICLE" ]]; then
  VEHICLE="${PROTOTYPE_ROOT}/config/vehicles/tvc_ring3_g12.json"
fi

# Default: primary landing log + F10 alternative + hover smoke
LOGS=(
  "$LOG_DIR/real_landing_g12.csv"
  "$LOG_DIR/real_landing_f10.csv"
  "$LOG_DIR/hover_hold_const.csv"
)

echo "=== Export ULog ==="
for csv in "${LOGS[@]}"; do
  if [[ -f "$csv" ]]; then
    ulg="${csv%.csv}.ulg"
    "$PY_SYS" "$ROOT/tools/csv_to_ulog.py" "$csv" -o "$ulg" --verify
  fi
done

echo ""
echo "=== Prototype 3D replay ==="
echo "Opening interactive viewers (close each window to continue)..."
for csv in "${LOGS[@]}"; do
  if [[ ! -f "$csv" ]]; then
    continue
  fi
  echo "--- replay $csv ---"
  # Prototype replay defaults to 3D when --view-3d is set
  (
    cd "$PROTOTYPE_ROOT"
    "$PY_PROTO" flight_test.py replay \
      --log "$csv" \
      --vehicle "$VEHICLE" \
      --view-3d
  ) || echo "viewer exited non-zero for $csv (window closed?)"
done

echo ""
echo "Done."
echo "ULog files:"
ls -la "$LOG_DIR"/*.ulg 2>/dev/null || true
