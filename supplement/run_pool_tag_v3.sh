#!/bin/bash
# Queue t50000_v3 tagging after the t50000 batch finishes (same node, avoid oversubscribing 242 cores).
PY=/weka/oe-training-default/jasonr/chenhao_prior/home/.conda/envs/trainer/bin/python
S=/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/CyclingCaptions/supplement/pool_tag.py
OUT=/weka/oe-training-default/chenhaoz/datacomp12b_supplement
until [ -f $OUT/logs/pool_tag.ALLDONE ]; do sleep 20; done
seq 0 99 | xargs -P 24 -I{} sh -c "$PY -u $S --src t50000_v3 --part {} --out $OUT --procs 8 > $OUT/logs/pool_tag_v3_{}.log 2>&1"
echo ALLDONE > $OUT/logs/pool_tag_v3.ALLDONE
