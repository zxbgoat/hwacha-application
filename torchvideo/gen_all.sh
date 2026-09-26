#!/bin/bash
# Generate every case export_tvid.py lists that has no <case>/<case>_tv.s yet, 3 at a time (mlir-opt /
# hwacha-mlir hold the constant-folded weights in memory: the largest, slowfast_r101, is 63M params).
# Logs in .logs/<case>.log, summary in .logs/summary.txt.
cd "$(dirname "$0")"; mkdir -p .logs
PYTHON=${PYTHON:-$HOME/hwacha-application/.tmenv/bin/python}
one() {
  n=$1; if [ -f "$n/${n}_tv.s" ]; then echo "$n SKIP (exists)"; return; fi
  if ./gen.sh "$n" >.logs/$n.log 2>&1; then echo "$n OK $(tail -1 .logs/$n.log)"
  else echo "$n FAIL: $(grep -m1 -oE 'LLVM ERROR[^\n]*|Error[^\n]*|error[^\n]*' .logs/$n.log | head -1)"; fi
}
export -f one
$PYTHON export_tvid.py --list 2>/dev/null | xargs -P3 -I{} bash -c 'one {}' | tee .logs/summary.txt
echo "==== $(grep -c ' OK ' .logs/summary.txt) ok, $(grep -c ' FAIL' .logs/summary.txt) fail, $(grep -c ' SKIP' .logs/summary.txt) skip"
