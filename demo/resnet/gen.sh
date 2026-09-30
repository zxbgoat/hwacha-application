#!/bin/bash
# Regenerate the demo from PyTorch: gen.sh [image]  (default dog.jpg)
# export.py (pretrained resnet50 + preprocessed image) -> mlir-opt bufferize -> hwacha-mlir --weights-bin
# -> hwacha-cc. Writes resnet50.s, resnet50_weights.bin(.S), check.bin, labels.h here.
set -eo pipefail
ROOT=${ROOT:-$HOME/hwacha-compiler}
MLIROPT=${MLIROPT:-$HOME/miniforge3/bin/mlir-opt}
PYTHON=${PYTHON:-$HOME/hwacha-application/.tmenv/bin/python}
HWMLIR=$ROOT/hwacha-cc/build/hwacha-mlir
HWCC=$ROOT/hwacha-cc/build/hwacha-cc
BUFFERIZE='builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},canonicalize,cse,ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,bufferization-lower-deallocations,cse,canonicalize,convert-bufferization-to-memref)'
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=$(realpath "${1:-$HERE/dog.jpg}")
W=$HERE/.build; rm -rf "$W"; mkdir -p "$W"; cd "$W"
T() { /usr/bin/time -f "[$1 maxrss=%MKB %es]" "${@:2}"; }
T export $PYTHON "$HERE/export.py" "$IMG" resnet50.mlir check.bin labels.h
T mlir-opt $MLIROPT resnet50.mlir --pass-pipeline="$BUFFERIZE" -o resnet50_memref.mlir
rm -f resnet50.mlir
T hwacha-mlir $HWMLIR resnet50_memref.mlir -o resnet50.ll --host resnet50.host.ll --weights-bin resnet50_weights.bin
T hwacha-cc $HWCC resnet50.ll --host resnet50.host.ll --assume-noalias -o resnet50.s
mv resnet50.s resnet50_weights.bin resnet50_weights.bin.S check.bin labels.h "$HERE/"
cp "$HERE/../../torchvision/hwlib.s" "$HERE/"
"$HERE/../../torchvision/split_weights.py" "$HERE/resnet50_weights.bin.S"
cd "$HERE"; rm -rf "$W"
echo "done: $(du -h resnet50_weights.bin | cut -f1) weights, $(du -h resnet50.s | cut -f1) asm"
