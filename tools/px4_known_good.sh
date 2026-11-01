#!/usr/bin/env bash
# Record / verify the last known-good Cube Orange+ .px4 (FLASH is ~98% full).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${BOARD:-cubepilot_cubeorangeplus_default}"
FLASH_LIMIT_BYTES="${FLASH_LIMIT_BYTES:-$((1920 * 1024))}"
KNOWN_GOOD="$ROOT/firmware/${BOARD}.px4.sha256"
ARTIFACTS="$ROOT/build/artifacts"
DEFAULT_PX4="$ARTIFACTS/${BOARD}.px4"

file_size() {
	if stat -f%z "$1" >/dev/null 2>&1; then
		stat -f%z "$1"
	else
		stat -c%s "$1"
	fi
}

sha256_hex() {
	if command -v shasum >/dev/null 2>&1; then
		shasum -a 256 "$1" | awk '{print $1}'
	else
		sha256sum "$1" | awk '{print $1}'
	fi
}

flash_pct() {
	python3 -c "print(f'{100.0 * $1 / $2:.2f}')"
}

known_good_hash() {
	local path="$1"
	if [[ ! -f "$path" ]]; then
		return 1
	fi
	awk 'NF && $1 !~ /^#/ {print $1; exit}' "$path"
}

summarize() {
	local px4="$1"
	local bin="${px4%.px4}.bin"
	local px4_size px4_hash
	px4_size="$(file_size "$px4")"
	px4_hash="$(sha256_hex "$px4")"
	echo "px4:  $px4"
	echo "size: $px4_size"
	echo "sha256: $px4_hash"
	if [[ -f "$bin" ]]; then
		local bin_size bin_hash pct
		bin_size="$(file_size "$bin")"
		bin_hash="$(sha256_hex "$bin")"
		pct="$(flash_pct "$bin_size" "$FLASH_LIMIT_BYTES")"
		echo "bin:  $bin"
		echo "bin_size: $bin_size"
		echo "bin_sha256: $bin_hash"
		echo "FLASH: $bin_size B / $FLASH_LIMIT_BYTES B (${pct}%)"
	fi
}

cmd_hash() {
	local px4="${1:-$DEFAULT_PX4}"
	if [[ ! -f "$px4" ]]; then
		echo "No .px4 at $px4" >&2
		exit 1
	fi
	summarize "$px4"
}

cmd_record() {
	local px4="${1:-$DEFAULT_PX4}"
	if [[ ! -f "$px4" ]]; then
		echo "No .px4 at $px4" >&2
		exit 1
	fi
	local px4_hash px4_size bin="${px4%.px4}.bin"
	px4_hash="$(sha256_hex "$px4")"
	px4_size="$(file_size "$px4")"
	mkdir -p "$(dirname "$KNOWN_GOOD")"
	{
		echo "# last known-good Cube Orange+ TV3 firmware"
		echo "# FLASH is ~98% full — rebuilds can fail silently; do not flash a mystery rebuild"
		echo "# recorded_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		echo "# px4_size=${px4_size}"
		if [[ -f "$bin" ]]; then
			local bin_size pct
			bin_size="$(file_size "$bin")"
			pct="$(flash_pct "$bin_size" "$FLASH_LIMIT_BYTES")"
			echo "# bin_size=${bin_size}"
			echo "# bin_sha256=$(sha256_hex "$bin")"
			echo "# FLASH=${bin_size}/${FLASH_LIMIT_BYTES} (${pct}%)"
		fi
		echo "${px4_hash}  ${BOARD}.px4"
	} >"$KNOWN_GOOD"
	echo "Recorded known-good SHA-256 → $KNOWN_GOOD"
	summarize "$px4"
}

cmd_verify() {
	local px4="${1:-$DEFAULT_PX4}"
	if [[ ! -f "$KNOWN_GOOD" ]]; then
		echo "No known-good hash at $KNOWN_GOOD" >&2
		exit 1
	fi
	if [[ ! -f "$px4" ]]; then
		echo "No .px4 at $px4" >&2
		exit 1
	fi
	local expected got
	expected="$(known_good_hash "$KNOWN_GOOD")"
	got="$(sha256_hex "$px4")"
	summarize "$px4"
	echo "known-good: $expected"
	if [[ "$got" == "$expected" ]]; then
		echo "MATCH last known-good .px4"
		return 0
	fi
	echo "MISMATCH — this is not the last known-good .px4" >&2
	return 1
}

usage() {
	cat <<EOF
Usage: $0 <hash|record|verify> [path/to.px4]

  hash    print SHA-256 and FLASH use
  record  write firmware/${BOARD}.px4.sha256 (last known-good)
  verify  check a .px4 against that known-good hash

Default image: $DEFAULT_PX4
EOF
}

cmd="${1:-}"
shift || true
case "$cmd" in
	hash) cmd_hash "${1:-}" ;;
	record) cmd_record "${1:-}" ;;
	verify) cmd_verify "${1:-}" ;;
	-h | --help | "") usage ;;
	*)
		echo "Unknown command: $cmd" >&2
		usage >&2
		exit 2
		;;
esac
