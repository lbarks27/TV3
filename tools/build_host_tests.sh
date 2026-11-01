#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build/host_tests"
cmake -S "$ROOT/tests/host" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
ctest --test-dir "$BUILD" --output-on-failure
echo "Host tests passed."
