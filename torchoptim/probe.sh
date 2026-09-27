#!/bin/bash
# probe.sh [case ...]: run export_opt.py for each case in its own process (torch-mlir can segfault), one line each
cd "$(dirname "$0")"; mkdir -p .logs/probe
P=$HOME/hwacha-application/.tmenv/bin/python
for n in ${@:-$($P export_opt.py --list)}; do
  if timeout 600 $P export_opt.py $n .logs/probe/$n.mlir .logs/probe/$n.bin > .logs/probe/$n.log 2>&1; then echo "$n OK $(grep -o 'in .* -> out .*' .logs/probe/$n.log)"
  else echo "$n FAIL($?) $(grep -m1 -oE "error: [^\"]{0,160}|Error[^\n]{0,160}|Exception[^\n]{0,160}|AssertionError[^\n]{0,120}" .logs/probe/$n.log | head -1)"; fi
  rm -f .logs/probe/$n.mlir .logs/probe/$n.bin
done
