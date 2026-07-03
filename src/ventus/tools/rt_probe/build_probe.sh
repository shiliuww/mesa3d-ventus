#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-/tmp/ventus_rt_probe_write_pixel.riscv}

riscv64-unknown-elf-gcc \
   -march=rv32g -mabi=ilp32 \
   -ffreestanding -fno-builtin -nostdlib -static \
   -T "${SCRIPT_DIR}/linker.ld" \
   "${SCRIPT_DIR}/probe_write_pixel.S" \
   -o "${OUT}"

echo "${OUT}"
readelf -h "${OUT}" | sed -n '1,12p'
