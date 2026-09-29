#!/bin/bash
# Regenerate one YOLO26 case from PyTorch: gen.sh <case> [HW] [hwacha-mlir flags]
# Writes <case>/{<case>.s, <case>_check.bin, <case>_weights.bin(.S), yolo_main.c, hwlib.s}.
# Retries with --collapse-all / --unroll-small=2 when hwacha-cc runs out of registers.
set -eo pipefail
ROOT=${ROOT:-$HOME/hwacha-compiler}
HWCCDIR=$ROOT/hwacha-cc
MLIROPT=${MLIROPT:-$HOME/miniforge3/bin/mlir-opt}
PYTHON=${PYTHON:-$HOME/hwacha-application/.tmenv/bin/python}
HWMLIR=$HWCCDIR/build/hwacha-mlir
HWCC=$HWCCDIR/build/hwacha-cc
BUFFERIZE='builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},canonicalize,cse,ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,bufferization-lower-deallocations,cse,canonicalize,convert-bufferization-to-memref)'
M=$1; HW=${2:-64}; FLAGS=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$HERE/.build/$M; rm -rf "$W"; mkdir -p "$W" "$HERE/$M"
cd "$W"
T() { /usr/bin/time -f "[$1 maxrss=%MKB %es]" "${@:2}"; }
T export $PYTHON "$HERE/export_yolo.py" $M $M.mlir ${M}_check.bin $HW
T mlir-opt $MLIROPT $M.mlir --pass-pipeline="$BUFFERIZE" -o ${M}_memref.mlir
rm -f $M.mlir
try() {  # $1 = hwacha-mlir flags
  T hwacha-mlir $HWMLIR ${M}_memref.mlir -o $M.ll --host $M.host.ll --weights-bin ${M}_weights.bin $1
  T hwacha-cc $HWCC $M.ll --host $M.host.ll --assume-noalias -o $M.s
}
rm -f "$HERE/$M/HWMLIRFLAGS"; [ -n "$FLAGS" ] && echo "$FLAGS" > "$HERE/$M/HWMLIRFLAGS"
if ! try "$FLAGS" 2>hwcc.err; then
  ok=0
  for extra in "--collapse-all" "--unroll-small=2" "--collapse-all --unroll-small=2"; do
    grep -q "out of Hwacha" hwcc.err || break
    [[ " $FLAGS " == *" $extra "* ]] && continue
    echo "$M: $(grep -o 'out of Hwacha[^"]*' hwcc.err | head -1) -> retry $extra"
    if try "$FLAGS $extra" 2>hwcc.err; then echo "$FLAGS $extra" | sed 's/^ *//' > "$HERE/$M/HWMLIRFLAGS"; ok=1; break; fi
  done
  [ $ok = 1 ] || { cat hwcc.err; exit 1; }
fi
grep -h 'maxrss' hwcc.err || true
mv $M.s ${M}_check.bin ${M}_weights.bin ${M}_weights.bin.S "$HERE/$M/"
cp "$HERE/yolo_main.c" "$HERE/hwlib.s" "$HERE/$M/"
"$HERE/split_weights.py" "$HERE/$M/${M}_weights.bin.S"
cd "$HERE"; rm -rf "$W"
echo "$M: done ($(du -h $M/${M}_weights.bin | cut -f1) weights, $(du -h $M/$M.s | cut -f1) asm)"
