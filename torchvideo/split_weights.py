#!/usr/bin/env python3
"""Rewrite <model>_tv_weights.bin.S so the first half of the blob (by bytes) lives in .weights_lo and the
second half in .weights_hi. tv.ld places .weights_lo before .text and .weights_hi after .bss, so every
%pcrel_hi reference from code stays within +-2 GB even when the weights alone exceed 2 GB. Idempotent."""
import re, sys
for path in sys.argv[1:]:
    lines = open(path).read().split('\n')
    if any('.weights_lo' in l for l in lines): continue
    ents = [(i, int(m.group(1)), int(m.group(2))) for i, l in enumerate(lines)
            if (m := re.match(r'\.incbin "[^"]+", (\d+), (\d+)', l))]
    if not ents: continue
    total = ents[-1][1] + ents[-1][2]
    # each entry starts with its .balign line 3 lines above the .incbin; split at the first entry past half
    cut = next((i for i, off, _ in ents if off >= total // 2), None)
    out, sect = [], None
    for i, l in enumerate(lines):
        if l.strip() == '.section .rodata': continue
        if l.startswith('.balign'):
            want = '.weights_hi' if (cut is not None and i >= cut - 3) else '.weights_lo'
            if want != sect: out.append(f'.section {want},"a",@progbits'); sect = want
        out.append(l)
    open(path, 'w').write('\n'.join(out))
    print(f'{path}: {total >> 20} MB, split at entry {ents.index(next(e for e in ents if e[0] == cut)) if cut else len(ents)}/{len(ents)}')
