"""Fine-grained class-name exposure in the two training pools.

Good runs (v0old, pretrain_131k) trained on metaclip_1b_merged; regressed runs
(v3_eng, v3_engnoneng, v4_eng) on the new curated v2_english pool. Both use the SAME
text target (k positives from original alt-text + dense sentences, prob 0.5), so if the
new pool's alt-text names fine-grained classes less often, that is the regression's source.
"""
import tarfile, json, re, os, glob, collections

E = "/weka/oe-training-default/oe-encoder/eval_data/clip_benchmark"
CB = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/CLIP_benchmark/clip_benchmark/datasets/en_classnames.json"
OUT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/evid/coverage2.json"
N = 8000

cn = json.load(open(CB))
cu = json.load(open(f"{E}/custom_classnames.json"))
food = [l.strip().replace("_", " ") for l in open(f"{E}/food101/food-101/meta/classes.txt") if l.strip()]
air = [l.strip() for l in open(f"{E}/fgvc_aircraft/fgvc-aircraft-2013b/data/variants.txt") if l.strip()]
VOCAB = {"flowers": cn["flowers"], "pets": cn["pets"], "food101": food,
         "fgvc_aircraft": air, "cars": cu["cars"], "sun397": cu["sun397"],
         "imagenet1k": cn["imagenet1k"]}
POOLS = {
    "new_v2_english": sorted(glob.glob("/weka/oe-mm-pretrain/oe-encoder/english_packed/0190/*.tar"))[:6],
    "metaclip_1b": sorted(glob.glob("/weka/oe-training-default/oe-encoder/metaclip_1b_merged/shard_0/w0/*.tar"))[:6],
}

norm = lambda s: " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())) + " "
PATS = {k: [re.compile(r"\b" + re.escape(norm(c).strip()) + r"\b") for c in v if len(c.strip()) > 2]
        for k, v in VOCAB.items()}
print({k: len(v) for k, v in PATS.items()}, flush=True)

res = {}
for pool, shards in POOLS.items():
    assert shards, pool
    cnt = collections.defaultdict(collections.Counter)
    n = 0
    chars = collections.Counter()
    for p in shards:
        if n >= N:
            break
        t = tarfile.open(p)
        for m in t:
            if not m.name.endswith(".json"):
                continue
            d = json.loads(t.extractfile(m).read())
            short = d.get("caption_short") or d.get("caption") or ""
            dense = d.get("caption_dense") or d.get("dense_caption") or ""
            ns, nd = norm(short), norm(dense)
            chars["short"] += len(short)
            chars["dense"] += len(dense)
            chars["n_dense_empty"] += (not dense)
            for k, ps in PATS.items():
                s = any(r.search(ns) for r in ps)
                dd = any(r.search(nd) for r in ps)
                cnt[k]["short"] += s
                cnt[k]["dense"] += dd
                cnt[k]["either"] += (s or dd)
                cnt[k]["short_only"] += (s and not dd)
            n += 1
            if n % 2000 == 0:
                print(f"  {pool} {n}", flush=True)
            if n >= N:
                break
        t.close()
    res[pool] = {"n": n, "mean_short_chars": round(chars["short"] / n, 1),
                 "mean_dense_chars": round(chars["dense"] / n, 1),
                 "pct": {k: {kk: round(100 * vv / n, 2) for kk, vv in c.items()} for k, c in cnt.items()}}
    print(pool, json.dumps(res[pool]), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", flush=True)
