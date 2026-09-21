#!/bin/bash
# run every case of a directory on Spike and keep the host's full result line: run_full.sh <dir>
d=$1; cd "$(dirname "$0")/../$d"; out=.logs/run_full.txt; mkdir -p .logs; : > $out
for c in $(make -s list); do
  make -s $c >/dev/null 2>&1 || { echo "$c BUILD-FAIL" >> $out; continue; }
  # every result line the host prints (the rodinia hosts print several), joined
  r=$(timeout 3600 make -s $c.spike 2>&1 | grep -E '^'"$c"'[:_ ]|max\|diff\||cycles|PASS|FAIL|mismatch|differ' | tr '\n' ' ')
  echo "$c ${r:-NO-OUTPUT}" >> $out
done; echo "DONE" >> $out
