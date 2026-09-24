#!/bin/bash
# Generate one case from PyTorch: gen.sh <fn> [hwacha-mlir flags]
# Writes <fn>/{<fn>.s, <fn>_check.bin, mod_main.c, hwlib.s (only if the assembly calls the library)}.
# Default mapping is --collapse-all (these single-op kernels overflow the scalar registers under the
# lane-innermost default); on "out of Hwacha registers" it retries the other mappings.
set -eo pipefail
ROOT=${ROOT:-/home/tesla/hwacha-compiler}; HWCCDIR=$ROOT/hwacha-cc
MLIROPT=${MLIROPT:-$HOME/miniforge3/bin/mlir-opt}; PYTHON=${PYTHON:-/home/tesla/hwacha-application/.tmenv/bin/python}
HWMLIR=$HWCCDIR/build/hwacha-mlir; HWCC=$HWCCDIR/build/hwacha-cc
BUFFERIZE='builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},canonicalize,cse,ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,bufferization-lower-deallocations,cse,canonicalize,convert-bufferization-to-memref)'
N=$1; FLAGS=${2---collapse-all}
HERE=$(cd "$(dirname "$0")" && pwd); W=$HERE/.build/$N; rm -rf "$W"; mkdir -p "$W" "$HERE/$N"; cd "$W"
$PYTHON "$HERE/export_intf.py" $N $N.mlir ${N}_check.bin
$MLIROPT $N.mlir --pass-pipeline="$BUFFERIZE" -o ${N}_memref.mlir
try() { $HWMLIR ${N}_memref.mlir -o $N.ll --host $N.host.ll $1 && $HWCC $N.ll --host $N.host.ll --assume-noalias -o $N.s; }
used="$FLAGS"
if ! try "$FLAGS" 2>hwcc.err; then
  ok=0
  for alt in "" "--collapse-all --unroll-small=2" "--unroll-small=2"; do
    grep -q "out of Hwacha" hwcc.err || break; [ "$alt" = "$FLAGS" ] && continue
    echo "$N: $(grep -o 'out of Hwacha[^"]*' hwcc.err | head -1) -> retry '${alt:-default mapping}'"
    if try "$alt" 2>hwcc.err; then used="$alt"; ok=1; break; fi
  done
  [ $ok = 1 ] || { cat hwcc.err; exit 1; }
fi
echo "$used" > "$HERE/$N/HWMLIRFLAGS"
mv $N.s ${N}_check.bin "$HERE/$N/"; cp "$HERE/mod_main.c" "$HERE/$N/"
syms=$(grep -oE '^[a-zA-Z_][a-zA-Z0-9_]*:' "$HERE/hwlib.s" | tr -d ':' | paste -sd'|')
if grep -qE "call[[:space:]]+($syms)\b" "$HERE/$N/$N.s"; then cp "$HERE/hwlib.s" "$HERE/$N/"; else rm -f "$HERE/$N/hwlib.s"; fi
cd "$HERE"; rm -rf "$W"; echo "$N: done ($(du -h $N/$N.s | cut -f1) asm, flags '$used')"
