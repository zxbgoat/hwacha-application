#!/bin/bash
# Generate every case export_tg.py lists that has no <case>/<case>.s yet, 4 at a time. Logs in
# .logs/<case>.log, summary in .logs/summary.txt.
cd "$(dirname "$0")"; mkdir -p .logs
PYTHON=${PYTHON:-$HOME/hwacha-application/.tmenv/bin/python}
one() {
  n=$1; if [ -f "$n/$n.s" ]; then echo "$n SKIP (exists)"; return; fi
  if timeout 1800 ./gen.sh "$n" >.logs/$n.log 2>&1; then echo "$n OK $(tail -1 .logs/$n.log)"
  else echo "$n FAIL: $(grep -m1 -oE 'error: [^\n]{0,120}|Error[^\n]{0,120}|out of Hwacha[^\n]{0,80}' .logs/$n.log | head -1)"; fi
}
export -f one
$PYTHON export_tg.py --list 2>/dev/null | xargs -P4 -I{} bash -c 'one {}' | tee .logs/summary.txt
echo "==== $(grep -c ' OK ' .logs/summary.txt) ok, $(grep -c ' FAIL' .logs/summary.txt) fail, $(grep -c ' SKIP' .logs/summary.txt) skip"
