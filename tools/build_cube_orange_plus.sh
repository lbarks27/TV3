#!/usr/bin/env bash
# Build TVC out-of-tree modules into PX4 for Cube Orange Plus.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Prefer a full PX4 clone; .work/px4-tv3 under this repo may be a broken worktree.
PX4_DIR="${PX4_DIR:-}"
px4_tree_ok() {
	[[ -f "$1/CMakeLists.txt" && -f "$1/platforms/nuttx/NuttX/nuttx/Makefile" ]]
}
if [[ -z "$PX4_DIR" ]]; then
	for candidate in \
		"$HOME/Developer/.work/px4-autopilot" \
		"$ROOT/../.work/px4-autopilot" \
		"$ROOT/../.work/px4-tv3" \
		"$HOME/Developer/.work/px4-tv3" \
		"$ROOT/.work/px4-tv3"; do
		if px4_tree_ok "$candidate"; then
			PX4_DIR="$candidate"
			break
		fi
	done
fi
if [[ -z "${PX4_DIR:-}" ]]; then
	PX4_DIR="$ROOT/.work/px4-tv3"
fi
BOARD="${BOARD:-cubepilot_cubeorangeplus_default}"

XPACK_GCC="$ROOT/.work/toolchains/xpack-arm-none-eabi-gcc-13.3.1-1.1/bin"
if [[ -x "$XPACK_GCC/arm-none-eabi-gcc" ]]; then
	export PATH="$XPACK_GCC:$PATH"
fi

if [[ ! -d "$PX4_DIR" ]]; then
	echo "PX4 tree not found at $PX4_DIR"
	echo "Set PX4_DIR or clone PX4-Autopilot (recommend v1.15.x) into .work/px4-tv3"
	exit 1
fi

# Install airframe into PX4 ROMFS (out-of-tree airframes need a copy)
AIRFRAME_SRC="$ROOT/ROMFS/init.d/airframes/15000_tvc_ring3"
AIRFRAME_DST="$PX4_DIR/ROMFS/px4fmu_common/init.d/airframes/15000_tvc_ring3"
cp "$AIRFRAME_SRC" "$AIRFRAME_DST"
chmod +x "$AIRFRAME_DST"
if [[ -f "${AIRFRAME_SRC}.post" ]]; then
	cp "${AIRFRAME_SRC}.post" "${AIRFRAME_DST}.post"
	chmod +x "${AIRFRAME_DST}.post"
fi

# Ensure airframes CMakeLists lists our files (idempotent append)
CMAKE_AIR="$PX4_DIR/ROMFS/px4fmu_common/init.d/airframes/CMakeLists.txt"
install_airframe_cmake() {
	local name="$1"
	if grep -q "$name" "$CMAKE_AIR"; then
		return 0
	fi
	if grep -q 'px4_add_romfs_files' "$CMAKE_AIR"; then
		perl -i -0pe "s/(\\n\\))/\\n\\t${name}\$1/s" "$CMAKE_AIR" || true
	fi
	if ! grep -q "$name" "$CMAKE_AIR"; then
		echo "WARNING: add $name to $CMAKE_AIR manually if missing"
	fi
}
install_airframe_cmake '15000_tvc_ring3'
if [[ -f "${AIRFRAME_DST}.post" ]]; then
	install_airframe_cmake '15000_tvc_ring3.post'
fi

# PX4 GLOB of *params.c only covers PX4_SOURCE_DIR/src, so out-of-tree
# params.c edits do not invalidate parameters.xml. Force a regen.
BOARD_BUILD="$PX4_DIR/build/${BOARD}"
GEN_XML="$BOARD_BUILD/parameters.xml"
if [[ -f "$GEN_XML" ]] && find "$ROOT/src/modules" -name '*params.c' -newer "$GEN_XML" | grep -q .; then
	echo "TV3 params.c newer than parameters.xml — regenerating"
	rm -f "$GEN_XML" "$BOARD_BUILD/src/lib/parameters/px4_parameters.hpp"
fi

echo "Building $BOARD with EXTERNAL_MODULES_LOCATION=$ROOT"
(
	cd "$PX4_DIR"
	make "$BOARD" EXTERNAL_MODULES_LOCATION="$ROOT"
)

BOARD_BUILD="$PX4_DIR/build/${BOARD}"
PX4_IMG="$BOARD_BUILD/${BOARD}.px4"
if [[ ! -f "$PX4_IMG" ]]; then
	# PX4 sometimes names the image without repeating the board stem
	if [[ -f "$BOARD_BUILD/cubepilot_cubeorangeplus_default.px4" ]]; then
		PX4_IMG="$BOARD_BUILD/cubepilot_cubeorangeplus_default.px4"
	fi
fi
if [[ ! -f "$PX4_IMG" ]]; then
	echo "Build finished but no .px4 under $BOARD_BUILD" >&2
	exit 1
fi

ARTIFACTS="$ROOT/build/artifacts"
mkdir -p "$ARTIFACTS"
cp "$PX4_IMG" "$ARTIFACTS/"
BIN_IMG="${PX4_IMG%.px4}.bin"
ELF_IMG="${PX4_IMG%.px4}.elf"
[[ -f "$BIN_IMG" ]] && cp "$BIN_IMG" "$ARTIFACTS/"
[[ -f "$ELF_IMG" ]] && cp "$ELF_IMG" "$ARTIFACTS/"

STAGED="$ARTIFACTS/$(basename "$PX4_IMG")"
HASH_OUT="${STAGED}.sha256"
if command -v shasum >/dev/null 2>&1; then
	shasum -a 256 "$STAGED" | tee "$HASH_OUT"
else
	sha256sum "$STAGED" | tee "$HASH_OUT"
fi

echo "Firmware: $STAGED"
if "$ROOT/tools/px4_known_good.sh" verify "$STAGED"; then
	:
else
	echo "This rebuild is not the last known-good image."
	echo "If it linked and you intend to fly it: $ROOT/tools/px4_known_good.sh record $STAGED"
fi
if [[ "${RECORD_KNOWN_GOOD:-0}" == "1" ]]; then
	"$ROOT/tools/px4_known_good.sh" record "$STAGED"
fi
