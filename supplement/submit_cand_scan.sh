#!/bin/bash
# Mass candidate scan: N CPU jobs, each handles shards i, i+N, i+2N, ... (round-robin over 1000),
# optionally only the first MAXSHARDS of the sequence. Resumable per shard (cand_scan.py skips done).
# usage: bash submit_cand_scan.sh <num_jobs> [max_shards_per_job] [tag]
N=${1:-180}; MAX=${2:-0}; TAG=${3:-cs}; REV=${4:-0}
export BEAKER_CONFIG=/weka/oe-training-default/jasonr/home/.beaker/config.yml
PY=/weka/oe-training-default/jasonr/chenhao_prior/home/.conda/envs/trainer/bin/python
SUB=/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/CyclingCaptions/pipeline_v3/submit_phase5.py
OUT=/weka/oe-training-default/chenhaoz/datacomp12b_supplement
# shard list per job is computed inside the job from {i} and N (keeps the --cmd short)
$PY -u $SUB --jobs $N --gpus 0 --preemptible --tag $TAG \
  --cmd "-c 'import subprocess,sys; i={i}; N=$N; M=$MAX; ks=[k for k in range(i,1000,N)]; ks=ks[:M] if M else ks; sys.exit(subprocess.call([\"$PY\",\"-u\",\"CyclingCaptions/supplement/cand_scan.py\",\"--shards\",\",\".join(map(str,ks)),\"--out\",\"$OUT\",\"--procs\",\"8\"]+([\"--reverse\"] if \"$REV\"==\"1\" else [])))'"
