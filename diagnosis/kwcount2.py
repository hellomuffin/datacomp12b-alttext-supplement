"""Second pass: fix two measurement gaps from kwcount.py.

1. Cars never matched: its 196 class names embed model years ("Acura TL Sedan 2012"),
   which no caption repeats verbatim. Here each car class is also matched by its
   make+model prefix ("acura tl"), and aircraft by its variant with separators
   normalised ("737-800" -> "737 800").
2. Dense rates were per whole dense caption (~198 words, ~10 sentences), but training
   samples ONE dense sentence per positive slot. So this also counts the per-sentence
   hit rate, which is what the text tower actually sees.

Writes one gz JSON per tar to kw2/<pool>/.
"""
import argparse, collections, gzip, json, os, re, tarfile
from multiprocessing import Pool

import sys
sys.path.insert(0, "/weka/oe-mm-pretrain/oe-encoder/_orchestrator")
from kwcount import vocabs, toks, new_tars, metaclip_tars, sidecar_for, MAX_NGRAM

OUTROOT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/kw2"
_sent = re.compile(r"[.!?\n]+")


def relaxed_vocabs():
    V = vocabs()
    out = {}
    # cars: strip a trailing 4-digit year, keep make+model (e.g. "Acura TL Sedan 2012" -> "acura tl sedan")
    out["cars_full"] = V["cars"]
    out["cars_noyear"] = [re.sub(r"\s*(19|20)\d{2}\s*$", "", c).strip() for c in V["cars"]]
    out["cars_make_model"] = [" ".join(toks(c)[:2]) for c in V["cars"]]
    # aircraft: "737-800" and "A340-300" tokenise to separate tokens already; keep as is
    out["fgvc_aircraft"] = V["fgvc_aircraft"]
    for k in ("flowers", "pets", "food101", "imagenet1k", "sun397"):
        out[k] = V[k]
    return out


def build_index(V):
    idx = collections.defaultdict(list)
    for v, classes in V.items():
        for c in classes:
            t = tuple(toks(c))
            if not t or len(t) > MAX_NGRAM:
                continue
            idx[t].append(f"{v}::{c}")
    return dict(idx)


def hits(tk, idx):
    out = set()
    n = len(tk)
    for i in range(n):
        for L in range(1, min(MAX_NGRAM, n - i) + 1):
            lab = idx.get(tuple(tk[i:i + L]))
            if lab:
                out.update(lab)
    return out


def scan(job):
    pool, tar = job
    idx = build_index(relaxed_vocabs())
    cls_alt = collections.Counter()
    cls_dense_cap = collections.Counter()   # class appears anywhere in the dense caption
    cls_dense_sent = collections.Counter()  # summed over sentences: #sentences naming the class
    st = collections.Counter()
    dense_map = None
    if pool == "new":
        dense_map = json.load(gzip.open(sidecar_for(tar), "rt"))["captions"]
    t = tarfile.open(tar)
    for m in t:
        if not m.name.endswith(".json"):
            continue
        d = json.loads(t.extractfile(m).read())
        alt = d.get("caption") or ""
        if pool == "new":
            dense = (dense_map.get(os.path.splitext(m.name)[0]) or {}).get("dense") or ""
        else:
            dense = d.get("dense_caption") or ""
        cls_alt.update(hits(toks(alt), idx))
        cls_dense_cap.update(hits(toks(dense), idx))
        sents = [s for s in _sent.split(dense) if s.strip()]
        st["n"] += 1
        st["n_sentences"] += len(sents)
        for s in sents:
            cls_dense_sent.update(hits(toks(s), idx))
    t.close()
    return dict(tar=tar, stats=dict(st), cls_alt=dict(cls_alt),
                cls_dense_cap=dict(cls_dense_cap), cls_dense_sent=dict(cls_dense_sent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=["new", "metaclip"])
    ap.add_argument("--tars", type=int, default=80)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    paths = new_tars(420)[:a.tars] if a.pool == "new" else metaclip_tars(500)[:a.tars]
    os.makedirs(f"{OUTROOT}/{a.pool}", exist_ok=True)

    def out_for(tar):
        rel = tar.replace("/weka/oe-training-default/oe-encoder/metaclip_1b_merged/", "").replace(
            "/weka/oe-mm-pretrain/oe-encoder/english_packed/", "")
        return f"{OUTROOT}/{a.pool}/" + rel.replace("/", "_").replace(".tar", ".json.gz")

    todo = [p for p in paths if not os.path.exists(out_for(p))]
    print(f"{a.pool}: {len(paths)} tars, {len(todo)} to do", flush=True)
    done = 0
    with Pool(a.procs) as pool_:
        for res in pool_.imap_unordered(scan, [(a.pool, p) for p in todo]):
            with gzip.open(out_for(res["tar"]), "wt") as f:
                json.dump(res, f)
            done += 1
            print(f"  [{done}/{len(todo)}] {res['stats']['n']} samples, "
                  f"{res['stats']['n_sentences']} sentences <- {res['tar']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
