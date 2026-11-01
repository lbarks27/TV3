#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PX4_DIR="${PX4_DIR:-}"
if [[ -z "$PX4_DIR" ]]; then
	for candidate in \
		"$ROOT/../.work/px4-tv3" \
		"$HOME/Developer/.work/px4-tv3" \
		"$ROOT/.work/px4-tv3"; do
		if [[ -f "$candidate/CMakeLists.txt" ]]; then
			PX4_DIR="$candidate"
			break
		fi
	done
fi
if [[ -z "${PX4_DIR:-}" || ! -d "$PX4_DIR" ]]; then
	echo "PX4 tree not found. Set PX4_DIR to a PX4-Autopilot checkout (v1.15.x recommended)."
	exit 1
fi
echo "Using PX4_DIR=$PX4_DIR"
# Prefer an out-of-tree build dir under this repo
BUILD_DIR="${BUILD_DIR:-$ROOT/build/px4_sitl_tvc}"
cmake -S "$PX4_DIR" -B "$BUILD_DIR" -DCONFIG=px4_sitl_default -DEXTERNAL_MODULES_LOCATION="$ROOT"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
