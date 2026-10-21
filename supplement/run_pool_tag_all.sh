#!/bin/bash
# Tag all 100 t50000 metadata parts locally: 24 parts at a time x 8 procs each (242-core node).
PY=/weka/oe-training-default/jasonr/chenhao_prior/home/.conda/envs/trainer/bin/python
S=/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/CyclingCaptions/supplement/pool_tag.py
OUT=/weka/oe-training-default/chenhaoz/datacomp12b_supplement
mkdir -p $OUT/logs
seq 0 99 | xargs -P 24 -I{} sh -c "$PY -u $S --part {} --out $OUT --procs 8 > $OUT/logs/pool_tag_{}.log 2>&1"
echo ALLDONE > $OUT/logs/pool_tag.ALLDONE
