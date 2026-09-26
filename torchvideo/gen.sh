#!/bin/bash
# Regenerate one pytorchvideo case from PyTorch: gen.sh <case> [hwacha-mlir flags]
# Writes <case>/{<case>_tv.s, <case>_tv_check.bin, <case>_tv_weights.bin(.S), tv_main.c, hwlib.s}.
# Retries with --collapse-all / --unroll-small=2 when hwacha-cc runs out of registers (3-D convs).
set -eo pipefail
ROOT=${ROOT:-$HOME/hwacha-compiler}
HWCCDIR=$ROOT/hwacha-cc
MLIROPT=${MLIROPT:-$HOME/miniforge3/bin/mlir-opt}
PYTHON=${PYTHON:-$HOME/hwacha-application/.tmenv/bin/python}
HWMLIR=$HWCCDIR/build/hwacha-mlir
HWCC=$HWCCDIR/build/hwacha-cc
BUFFERIZE='builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},canonicalize,cse,ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,bufferization-lower-deallocations,cse,canonicalize,convert-bufferization-to-memref)'
M=$1; FLAGS=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$HERE/.build/$M; rm -rf "$W"; mkdir -p "$W" "$HERE/$M"
cd "$W"
T() { /usr/bin/time -f "[$1 maxrss=%MKB %es]" "${@:2}"; }
T export $PYTHON "$HERE/export_tvid.py" $M ${M}_tv.mlir ${M}_tv_check.bin
T mlir-opt $MLIROPT ${M}_tv.mlir --pass-pipeline="$BUFFERIZE" -o ${M}_tv_memref.mlir
rm -f ${M}_tv.mlir
try() {  # $1 = hwacha-mlir flags
  T hwacha-mlir $HWMLIR ${M}_tv_memref.mlir -o ${M}_tv.ll --host ${M}_tv.host.ll --weights-bin ${M}_tv_weights.bin $1
  T hwacha-cc $HWCC ${M}_tv.ll --host ${M}_tv.host.ll --assume-noalias -o ${M}_tv.s
}
echo "$FLAGS" | sed 's/^ *//' > "$HERE/$M/HWMLIRFLAGS"
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
mv ${M}_tv.s ${M}_tv_check.bin ${M}_tv_weights.bin ${M}_tv_weights.bin.S "$HERE/$M/"
cp "$HERE/tv_main.c" "$HERE/hwlib.s" "$HERE/$M/"
"$HERE/split_weights.py" "$HERE/$M/${M}_tv_weights.bin.S"
cd "$HERE"; rm -rf "$W"
echo "$M: done ($(du -h $M/${M}_tv_weights.bin | cut -f1) weights, $(du -h $M/${M}_tv.s | cut -f1) asm)"
