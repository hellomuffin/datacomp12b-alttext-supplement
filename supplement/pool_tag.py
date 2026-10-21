"""Supplement step 3a: per-entry counts of the CURRENT pool, split by alt-text quality.

Input : datacomp12b_english_selected_v2/t50000_metadata/part_{i}.parquet (uid, url, text)
Output: <out>/pool/part_{i}.npz with
          counts_all   [500000] int64  entry hits over all captions (sanity vs post_selection_dist)
          counts_inf   [500000] int64  entry hits over INFORMATIVE captions only
          n, n_inf, n_dup, n_placeholder, n_short
Matching = MetaCLIP substring matching (spaced " entry " in spaced text) via Aho-Corasick,
identical semantics to reference_code/MetaCLIP/metaclip/metaclip1/substr_matching_ahocorasick.py.
"""
import argparse, os, sys, time, json, hashlib
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality import is_informative, spacing, build_automaton, NUM_ENTRIES

META = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/reference_code/MetaCLIP/metaclip/metaclip1/metadata.json"
POOLS = {"t50000": "/weka/oe-training-default/chenhaoz/datacomp12b_english_selected_v2/t50000_metadata",
         "t50000_v3": "/weka/oe-training-default/chenhaoz/datacomp12b_english_selected_v2/t50000_v3/metadata"}
_A = None


def _init():
    global _A
    _A = build_automaton(json.load(open(META)))


def _chunk(texts):
    ca = np.zeros(NUM_ENTRIES, np.int64)
    ci = np.zeros(NUM_ENTRIES, np.int64)
    st = np.zeros(5, np.int64)  # n, n_inf, n_dup, n_placeholder, n_short
    seen = set()
    for t in texts:
        t = t or ""
        st[0] += 1
        h = hashlib.md5(t.strip().lower().encode()).digest()[:8]
        dup = h in seen
        seen.add(h)
        ok, why = is_informative(t)
        ids = list({v for _, v in _A.iter(spacing(t))})
        if ids:
            ca[ids] += 1
        if dup:
            st[2] += 1
            continue
        if not ok:
            st[3 if why == "placeholder" else 4] += 1
            continue
        st[1] += 1
        if ids:
            ci[ids] += 1
    return ca, ci, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="rows (0=all) for smoke tests")
    ap.add_argument("--src", default="t50000", choices=list(POOLS), help="which selection's metadata to tag")
    a = ap.parse_args()
    out = f"{a.out}/pool/part_{a.part}.npz" if a.src == "t50000" else f"{a.out}/pool_{a.src}/part_{a.part}.npz"
    if os.path.exists(out):
        print("exists, skip", out, flush=True)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    tab = pq.read_table(f"{POOLS[a.src]}/part_{a.part}.parquet", columns=["text"])
    texts = tab.column("text").to_pylist()
    if a.limit:
        texts = texts[:a.limit]
    print(f"part {a.part}: {len(texts):,} rows read in {time.time()-t0:.0f}s", flush=True)
    chunks = [texts[i::a.procs] for i in range(a.procs)]
    with Pool(a.procs, initializer=_init) as p:
        res = p.map(_chunk, chunks)
    ca = sum(r[0] for r in res)
    ci = sum(r[1] for r in res)
    st = sum(r[2] for r in res)
    np.savez(out, counts_all=ca, counts_inf=ci, n=st[0], n_inf=st[1], n_dup=st[2],
             n_placeholder=st[3], n_short=st[4])
    print(f"part {a.part}: n={st[0]:,} informative={st[1]:,} ({100*st[1]/st[0]:.1f}%) dup={st[2]:,} "
          f"placeholder={st[3]:,} short={st[4]:,} | entry-hits all={ca.sum():,} inf={ci.sum():,} | "
          f"{time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
