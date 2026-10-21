"""Class-name coverage of the NEW pool's DENSE captions (held in gzip side-cars,
not inside the tars) so they can be compared with metaclip's inline dense_caption
numbers from coverage2.py."""
import gzip, json, re, glob, collections

E = "/weka/oe-training-default/oe-encoder/eval_data/clip_benchmark"
CB = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/CLIP_benchmark/clip_benchmark/datasets/en_classnames.json"
OUT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/evid/coverage3.json"
N = 8000

cn = json.load(open(CB))
cu = json.load(open(f"{E}/custom_classnames.json"))
food = [l.strip().replace("_", " ") for l in open(f"{E}/food101/food-101/meta/classes.txt") if l.strip()]
air = [l.strip() for l in open(f"{E}/fgvc_aircraft/fgvc-aircraft-2013b/data/variants.txt") if l.strip()]
VOCAB = {"flowers": cn["flowers"], "pets": cn["pets"], "food101": food,
         "fgvc_aircraft": air, "cars": cu["cars"], "sun397": cu["sun397"],
         "imagenet1k": cn["imagenet1k"]}
norm = lambda s: " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())) + " "
PATS = {k: [re.compile(r"\b" + re.escape(norm(c).strip()) + r"\b") for c in v if len(c.strip()) > 2]
        for k, v in VOCAB.items()}

cnt = collections.defaultdict(collections.Counter)
n = 0
chars = 0
for p in sorted(glob.glob("/weka/oe-mm-pretrain/oe-encoder/english_captions/0190/*.captions.json.gz"))[:2]:
    if n >= N:
        break
    caps = json.load(gzip.open(p, "rt"))["captions"]
    for key, v in caps.items():
        dense = v.get("dense") or ""
        nd = norm(dense)
        chars += len(dense)
        for k, ps in PATS.items():
            cnt[k]["dense"] += any(r.search(nd) for r in ps)
        n += 1
        if n % 2000 == 0:
            print(f"  {n}", flush=True)
        if n >= N:
            break

res = {"n": n, "mean_dense_chars": round(chars / n, 1),
       "pct": {k: {kk: round(100 * vv / n, 2) for kk, vv in c.items()} for k, c in cnt.items()}}
json.dump(res, open(OUT, "w"), indent=1)
print("new_v2_english_dense", json.dumps(res), flush=True)
print("DONE", flush=True)
