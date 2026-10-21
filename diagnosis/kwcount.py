"""Keyword distribution of the two pretraining pools, counted (not sampled by eye).

Pools
  new      = the v3/v4 curated pool: /weka/oe-mm-pretrain/oe-encoder/english_packed/**.tar
             alt-text lives in the tar's per-sample .json ("caption"); the dense caption
             lives in a gzip side-car under english_captions/ keyed by sample name.
  metaclip = the v0old/pretrain_131k pool: metaclip_1b_merged/shard_*/w*/*.tar
             with "caption" and "dense_caption" both inline in the .json.

Per shard this writes one gz JSON holding
  * full unigram counts for alt-text and for dense captions (for JS divergence and
    for any token-level statistic computed later),
  * per-class exact-phrase counts for every benchmark vocabulary, separately for
    alt-text and dense captions, by n-gram lookup (1..6 word phrases) so the cost is
    O(words) per caption instead of O(#classes),
  * caption length sums/histograms and duplicate-caption counts.

Resumable: a shard whose output file already exists is skipped.
"""
import argparse, collections, gzip, hashlib, json, os, re, sys, tarfile
from multiprocessing import Pool

E = "/weka/oe-training-default/oe-encoder/eval_data/clip_benchmark"
CB = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/CLIP_benchmark/clip_benchmark/datasets/en_classnames.json"
MANIFEST = "/weka/oe-training-default/oe-encoder/manifests/v2_english-20260904.jsonl"
METACLIP = "/weka/oe-training-default/oe-encoder/metaclip_1b_merged"
OUTROOT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/kw"
MAX_NGRAM = 6


def vocabs():
    cn = json.load(open(CB))
    cu = json.load(open(f"{E}/custom_classnames.json"))
    food = [l.strip().replace("_", " ") for l in open(f"{E}/food101/food-101/meta/classes.txt") if l.strip()]
    air = [l.strip() for l in open(f"{E}/fgvc_aircraft/fgvc-aircraft-2013b/data/variants.txt") if l.strip()]
    return {"flowers": cn["flowers"], "pets": cn["pets"], "food101": food,
            "fgvc_aircraft": air, "cars": cu["cars"], "sun397": cu["sun397"],
            "imagenet1k": cn["imagenet1k"]}


_word = re.compile(r"[a-z0-9]+")


def toks(s):
    return _word.findall(s.lower())


def build_index(V):
    """phrase tuple -> list of 'vocab::class' labels."""
    idx = collections.defaultdict(list)
    for v, classes in V.items():
        for c in classes:
            t = tuple(toks(c))
            if not t or len(t) > MAX_NGRAM:
                continue
            idx[t].append(f"{v}::{c}")
    return dict(idx)


def hits(tk, idx):
    """Every class label whose full name appears as a phrase in this token list."""
    out = set()
    n = len(tk)
    for i in range(n):
        for L in range(1, min(MAX_NGRAM, n - i) + 1):
            lab = idx.get(tuple(tk[i:i + L]))
            if lab:
                out.update(lab)
    return out


def out_name(pool, tar_path):
    """Unique per tar: metaclip basenames repeat across shards (shard_7/w3/00000000.tar)."""
    rel = tar_path.replace(METACLIP + "/", "").replace(
        "/weka/oe-mm-pretrain/oe-encoder/english_packed/", "")
    return os.path.join(OUTROOT, pool, rel.replace("/", "_").replace(".tar", ".json.gz"))


def sidecar_for(tar_path):
    p = tar_path.replace("_packed", "_captions")
    return os.path.splitext(p)[0] + ".captions.json.gz"


def scan_tar(job):
    pool, tar_path = job
    uni_a, uni_d = collections.Counter(), collections.Counter()
    cls_a, cls_d = collections.Counter(), collections.Counter()
    dup = collections.Counter()
    st = collections.Counter()
    V = build_index(vocabs())
    dense_map = None
    if pool == "new":
        sc = sidecar_for(tar_path)
        if not os.path.exists(sc):
            raise FileNotFoundError(sc)
        dense_map = json.load(gzip.open(sc, "rt"))["captions"]
    t = tarfile.open(tar_path)
    for m in t:
        if not m.name.endswith(".json"):
            continue
        d = json.loads(t.extractfile(m).read())
        alt = d.get("caption") or ""
        if pool == "new":
            key = os.path.splitext(m.name)[0]
            ent = dense_map.get(key)
            dense = (ent or {}).get("dense") or ""
        else:
            dense = d.get("dense_caption") or ""
        ta, td = toks(alt), toks(dense)
        uni_a.update(ta)
        uni_d.update(td)
        cls_a.update(hits(ta, V))
        cls_d.update(hits(td, V))
        st["n"] += 1
        st["alt_chars"] += len(alt)
        st["alt_words"] += len(ta)
        st["dense_chars"] += len(dense)
        st["dense_words"] += len(td)
        st["alt_empty"] += (not alt)
        st["dense_empty"] += (not dense)
        st["alt_has_digit"] += any(ch.isdigit() for ch in alt)
        dup[hashlib.md5(alt.strip().lower().encode()).hexdigest()[:16]] += 1
    t.close()
    st["dup_alt_captions"] = sum(v for v in dup.values() if v > 1)
    st["distinct_alt_captions"] = len(dup)
    return dict(stats=dict(st), uni_alt=dict(uni_a), uni_dense=dict(uni_d),
                cls_alt=dict(cls_a), cls_dense=dict(cls_d), tar=tar_path)


def new_tars(n_total):
    paths = []
    with open(MANIFEST) as f:
        for line in f:
            o = json.loads(line)
            if "path" in o:
                paths.append(o["path"])
    step = max(1, len(paths) // n_total)
    return paths[::step][:n_total]


def metaclip_tars(n_total, n_shards=40):
    """Spread over shards AND over the w0..w7 sub-shards, so the sample is not one writer."""
    shards = sorted(s for s in os.listdir(METACLIP) if s.startswith("shard_"))
    step = max(1, len(shards) // n_shards)
    chosen = shards[::step][:n_shards]
    per = max(1, n_total // max(1, len(chosen)))
    out = []
    for k, s in enumerate(chosen):
        wd = os.path.join(METACLIP, s, f"w{k % 8}")
        if not os.path.isdir(wd):
            continue
        tars = sorted(x for x in os.listdir(wd) if x.endswith(".tar"))
        out += [os.path.join(wd, x) for x in tars[:per]]
    return out[:n_total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-idx", type=int, required=True)
    ap.add_argument("--total-jobs", type=int, required=True)
    ap.add_argument("--new-tars", type=int, default=420)
    ap.add_argument("--metaclip-tars", type=int, default=500)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    jobs = [("new", p) for p in new_tars(a.new_tars)] + [("metaclip", p) for p in metaclip_tars(a.metaclip_tars)]
    mine = [j for k, j in enumerate(jobs) if k % a.total_jobs == a.job_idx]
    todo = []
    for pool, tar in mine:
        out = out_name(pool, tar)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not os.path.exists(out):
            todo.append((pool, tar, out))
    print(f"job {a.job_idx}/{a.total_jobs}: {len(mine)} tars assigned, {len(todo)} to do", flush=True)
    done = 0
    with Pool(a.procs) as pool_:
        # write each result under its OWN tar name (imap_unordered returns out of order)
        for res in pool_.imap_unordered(scan_tar, [(p, t) for p, t, _ in todo]):
            o = out_name("new" if "english_packed" in res["tar"] else "metaclip", res["tar"])
            with gzip.open(o, "wt") as f:
                json.dump(res, f)
            done += 1
            print(f"  [{done}/{len(todo)}] {res['stats']['n']} samples <- {res['tar']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
