#!/bin/bash
# Generate every case in models.txt that has no <case>/<case>.s yet: the n / s / m scales 3 at a time,
# l 2 at a time, x one by one (mlir-opt / hwacha-mlir hold the constant-folded weights in memory).
# Logs in .logs/<case>.log, summary in .logs/summary.txt.
cd "$(dirname "$0")"; mkdir -p .logs
one() {  # one case
  read -r n hw <<<"$(grep -E "^$1 " models.txt)"
  if [ -f "$n/$n.s" ]; then echo "$n SKIP (exists)"; return; fi
  if ./gen.sh "$n" "$hw" >.logs/$n.log 2>&1; then echo "$n OK $(tail -1 .logs/$n.log)"
  else echo "$n FAIL: $(grep -m1 -oE 'LLVM ERROR[^\n]*|Error[^\n]*|error[^\n]*' .logs/$n.log | head -1)"; fi
}
export -f one
tier() { grep -vE '^#|^$' models.txt | awk '{print $1}' | grep -E "^yolo(v8|26|11)[$1]"; }
{ tier nsm | xargs -P3 -I{} bash -c 'one {}'
  tier l   | xargs -P2 -I{} bash -c 'one {}'
  tier x   | xargs -P1 -I{} bash -c 'one {}'
} | tee .logs/summary.txt
echo "==== $(grep -c ' OK ' .logs/summary.txt) ok, $(grep -c ' FAIL' .logs/summary.txt) fail, $(grep -c ' SKIP' .logs/summary.txt) skip"
