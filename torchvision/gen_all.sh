#!/bin/bash
# Generate every model in models.txt that has no <model>/<model>_tv.s yet.
# Small models run 3 at a time, mid-size 2, the largest one by one (mlir-opt/hwacha-mlir hold the
# constant-folded weights in memory). Logs in .logs/<model>.log, summary in .logs/summary.txt.
cd "$(dirname "$0")"; mkdir -p .logs
PYTHON=${PYTHON:-/home/tesla/hwacha-application/.tmenv/bin/python}
# param counts (millions) -> tier
$PYTHON - <<'PY' > .logs/params.txt 2>/dev/null
import torchvision.models as M, warnings; warnings.simplefilter('ignore')
for l in open('models.txt'):
    if l.startswith('#') or not l.strip(): continue
    n=l.split()[0]
    m = M.maxvit.MaxVit(stem_channels=64, block_channels=[64,128,256,512], block_layers=[2,2,5,2], stochastic_depth_prob=0.2, head_dim=32, input_size=(64,64), partition_size=2) if n=='maxvit_t' else M.get_model(n, weights=None)
    print(n, round(sum(p.numel() for p in m.parameters())/1e6,1))
PY
one() {  # one model
  read -r n hw kw <<<"$(grep -E "^$1 " models.txt)"
  if [ -f "$n/${n}_tv.s" ]; then echo "$n SKIP (exists)"; return; fi
  if ./gen.sh "$n" "$hw" "$kw" >.logs/$n.log 2>&1; then echo "$n OK $(tail -1 .logs/$n.log)"
  else echo "$n FAIL: $(grep -m1 -oE 'LLVM ERROR[^\n]*|Error[^\n]*|error[^\n]*' .logs/$n.log | head -1)"; fi
}
export -f one
tier() { awk -v lo=$1 -v hi=$2 '$2>=lo && $2<hi {print $1}' .logs/params.txt; }
{ tier 0 60   | xargs -P3 -I{} bash -c 'one {}'
  tier 60 200 | xargs -P1 -I{} bash -c 'one {}'
  tier 200 9999 | xargs -P1 -I{} bash -c 'one {}'
} | tee .logs/summary.txt
echo "==== $(grep -c ' OK ' .logs/summary.txt) ok, $(grep -c ' FAIL' .logs/summary.txt) fail, $(grep -c ' SKIP' .logs/summary.txt) skip"
