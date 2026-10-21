"""Supplement steps 1-3 in ONE pass over the DataComp-12B English metadata on GCS.

The packed English pool is the DENSE-caption MetaCLIP selection
(/weka/oe-mm/chenhaoz/dense_metaclip_v1/selected/t105521/shard_{k}.pkl, compound keys
"{uid}_{key}_{sha}"), NOT the older alt-text selections. Per metadata shard k:

  pool rows  (uid in that shard's selected keys): tag alt-text -> counts_pool_all / counts_pool_inf
             (= the current pool's per-entry alt-text exposure, split by caption quality)
  candidates (English, not in pool, informative alt-text, not duplicate within shard):
             tag -> counts_cand_inf, and write uid/url/text/entries to parquet

Output per shard: <out>/cand/shard_{k}.parquet (candidates) and shard_{k}.npz (all counts+stats).
Resumable: skips shards whose .npz exists. Runs on the cluster (GCS creds from env).
"""
import argparse, os, sys, time, json, hashlib, pickle, tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality import is_informative, spacing, build_automaton, NUM_ENTRIES

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    _kf = os.path.join(tempfile.gettempdir(), "gcp-sa-key.json")
    with open(_kf, "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _kf
import certifi
for _v in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):  # container CA bundle is broken
    os.environ[_v] = certifi.where()

META = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/reference_code/MetaCLIP/metaclip/metaclip1/metadata.json"
GCS = "datacomp1b-metadata-dump/datacomp12B_metadata/1000_shards/shard_{k}"
ENG = "/weka/oe-training-default/chenhaoz/datacomp12b_english_uids_v2/shard_{k}.pkl"
POOL = "/weka/oe-mm/chenhaoz/dense_metaclip_v1/selected/t105521/shard_{k}.pkl"
_A = None


def _init():
    global _A
    _A = build_automaton(json.load(open(META)))


def _tag(texts):
    return [sorted({v for _, v in _A.iter(spacing(t))}) for t in texts]


def tag_all(texts, procs):
    chunks = [texts[i::procs] for i in range(procs)]
    with Pool(procs, initializer=_init) as p:
        parts = p.map(_tag, chunks)
    out = [None] * len(texts)
    for i, ch in enumerate(parts):
        out[i::procs] = ch
    return out


def counts_of(entries_list):
    c = np.zeros(NUM_ENTRIES, np.int64)
    for e in entries_list:
        if e:
            c[e] += 1
    return c


def process_rows(rows, eng, pool, procs):
    """rows: iterable of (uid, url, text). Returns (cand_table, npz_dict)."""
    st = dict(n_rows=0, n_english=0, n_pool=0, n_pool_inf=0, n_pool_dup=0, n_pool_placeholder=0,
              n_pool_short=0, n_pool_urlish=0, n_unused=0, n_dup=0, n_placeholder=0, n_short=0,
              n_urlish=0, n_kept=0)
    pool_texts, pool_inf = [], []
    uids, urls, texts = [], [], []
    seen_c, seen_p = set(), set()
    for uid, url, text in rows:
        st["n_rows"] += 1
        if uid not in eng:
            continue
        st["n_english"] += 1
        text = text or ""
        if uid in pool:
            st["n_pool"] += 1
            h = hashlib.md5(text.strip().lower().encode()).digest()[:8]
            ok, why = is_informative(text)
            inf = ok and h not in seen_p
            if h in seen_p:
                st["n_pool_dup"] += 1
            elif not ok:
                st["n_pool_" + why] += 1
            seen_p.add(h)
            st["n_pool_inf"] += inf
            pool_texts.append(text); pool_inf.append(inf)
            continue
        st["n_unused"] += 1
        ok, why = is_informative(text)
        if not ok:
            st["n_" + why] += 1
            continue
        h = hashlib.md5(text.strip().lower().encode()).digest()[:8]
        if h in seen_c:
            st["n_dup"] += 1
            continue
        seen_c.add(h)
        uids.append(uid); urls.append(url); texts.append(text)
    st["n_kept"] = len(uids)
    pool_entries = tag_all(pool_texts, procs) if pool_texts else []
    cand_entries = tag_all(texts, procs) if texts else []
    npz = dict(counts_cand_inf=counts_of(cand_entries),
               counts_pool_all=counts_of(pool_entries),
               counts_pool_inf=counts_of([e for e, i in zip(pool_entries, pool_inf) if i]),
               **{k: np.int64(v) for k, v in st.items()})
    tab = pa.table({"uid": pa.array(uids, pa.string()), "url": pa.array(urls, pa.string()),
                    "text": pa.array(texts, pa.string()),
                    "entries": pa.array(cand_entries, pa.list_(pa.int32()))})
    return tab, npz


def load_sets(k):
    eng = set(pickle.load(open(ENG.format(k=k), "rb")))
    pp = POOL.format(k=k)
    pool = {key.split("_", 1)[0] for key in pickle.load(open(pp, "rb"))} if os.path.exists(pp) else set()
    return eng, pool


def scan_shard(k, out, procs, fs):
    t0 = time.time()
    npz = f"{out}/cand/shard_{k}.npz"
    if os.path.exists(npz):
        print(f"shard {k}: exists, skip", flush=True)
        return
    import gcsfs  # noqa
    files = sorted(f for f in fs.ls(GCS.format(k=k)) if f.endswith(".parquet"))
    eng, pool = load_sets(k)

    def rows():
        for f in files:
            with fs.open(f, "rb") as fh:
                pf = pq.ParquetFile(fh)
                tcol = "text" if "text" in pf.schema_arrow.names else "caption"
                tab = pf.read(columns=["uid", "url", tcol])
            yield from zip(tab.column("uid").to_pylist(), tab.column("url").to_pylist(),
                           tab.column(tcol).to_pylist())

    tab, d = process_rows(rows(), eng, pool, procs)
    os.makedirs(f"{out}/cand", exist_ok=True)
    tmp = f"{out}/cand/shard_{k}.parquet.tmp"
    pq.write_table(tab, tmp, compression="zstd")
    os.replace(tmp, f"{out}/cand/shard_{k}.parquet")
    np.savez(npz, **d)
    st = {k2: int(v) for k2, v in d.items() if k2.startswith("n_")}
    print(f"shard {k}: rows={st['n_rows']:,} eng={st['n_english']:,} pool={st['n_pool']:,} "
          f"(pool inf {100*st['n_pool_inf']/max(1,st['n_pool']):.1f}%) unused={st['n_unused']:,} "
          f"kept={st['n_kept']:,} (placeholder={st['n_placeholder']:,} short={st['n_short']:,} "
          f"urlish={st['n_urlish']:,} dup={st['n_dup']:,}) | {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="comma list or a-b range")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--reverse", action="store_true", help="walk the shard list backwards (second wave meets the first in the middle)")
    a = ap.parse_args()
    if "-" in a.shards:
        lo, hi = a.shards.split("-"); ks = list(range(int(lo), int(hi)))
    else:
        ks = [int(x) for x in a.shards.split(",")]
    if a.reverse:
        ks = ks[::-1]
    import gcsfs
    fs = gcsfs.GCSFileSystem(timeout=300, retries=5)
    for k in ks:
        try:
            scan_shard(k, a.out, a.procs, fs)
        except Exception as e:
            print(f"shard {k}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
