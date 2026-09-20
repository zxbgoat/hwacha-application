#!/bin/bash
cd "$(dirname "$0")/.."
until grep -q '^DONE' torch-vision/.logs/run_full.txt 2>/dev/null; do sleep 60; done
/home/tesla/hwacha-application/.tmenv/bin/python tools/gen_case_readme.py torch-vision > .logs_vision_readme.out 2>&1
echo "READMES DONE $(date)" >> .logs_vision_readme.out
