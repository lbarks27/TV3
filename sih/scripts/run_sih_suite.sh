#!/usr/bin/env bash
# Run primary SIH acceptance flights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PROTOTYPE_ROOT="${PROTOTYPE_ROOT:-$ROOT/sim_source}"
PY="${PYTHON:-$PROTOTYPE_ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

cd "$ROOT"
echo "=== SIH suite (PROTOTYPE_ROOT=$PROTOTYPE_ROOT) ==="
fail=0
for profile in real_landing_g12 real_landing_f10 pad_hover_g12 pad_hop_c6 hover_hold_const; do
  echo "--- $profile ---"
  if ! "$PY" sih/run_sih.py --profile "$profile"; then
    fail=1
  else
    cp -f sih/logs/last_result.json "sih/logs/last_result_${profile}.json"
  fi
done
if [[ $fail -ne 0 ]]; then
  echo "SIH suite FAILED"
  exit 1
fi
echo "SIH suite PASSED"
cat sih/logs/last_result.json
