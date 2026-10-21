"""Supplement step 4: MetaCLIP-style balanced acceptance over the scanned candidates.

For each candidate with matched entries E(x): accept iff any entry j in E(x) passes
Bernoulli(p_j), where p_j = min(1, d_j / s_j) comes from deficit.py (deficit in place of t).
Optional --budget scales all p_j by a common factor lambda (bisection on expected fill) so the
expected number of accepted samples hits the budget.  Embarrassingly parallel per shard.

Output: <out>/selected_t{t}[_b{budget}]/shard_{k}.parquet (uid, url, text, entries) + summary.
"""
import argparse, glob, os, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool

_P = None


def _init(p):
    global _P
    _P = p


def _accept_mask(entries_col, seed):
    rng = np.random.default_rng(seed)
    offsets = entries_col.offsets.to_numpy()
    vals = entries_col.values.to_numpy()
    r = rng.random(len(vals))
    pass_ = r < _P[vals]
    # any() per row via reduceat on segment boundaries (handle empty rows)
    n = len(offsets) - 1
    out = np.zeros(n, bool)
    starts, ends = offsets[:-1], offsets[1:]
    nonempty = ends > starts
    if nonempty.any():
        seg = np.add.reduceat(pass_.astype(np.int32), starts[nonempty])
        out[nonempty] = seg > 0
    return out


def _one(args):
    path, outdir, seed = args
    k = os.path.basename(path).replace(".parquet", "")
    dst = f"{outdir}/{k}.parquet"
    if os.path.exists(dst):
        z = pq.read_metadata(dst).num_rows
        return k, None, z
    tab = pq.read_table(path)
    mask = np.zeros(tab.num_rows, bool)
    # chunked columns: process per chunk
    off = 0
    for chunk in tab.column("entries").chunks:
        m = _accept_mask(chunk, seed + off)
        mask[off:off + len(m)] = m
        off += len(m)
    sel = tab.filter(pa.array(mask))
    pq.write_table(sel, dst + ".tmp", compression="zstd")
    os.replace(dst + ".tmp", dst)
    return k, tab.num_rows, sel.num_rows


_SAMPLE_CACHE = None


def _load_sample(cand_files, sample=20):
    """Cache (offsets, values) of the sample files once; the bisection then costs pure numpy."""
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is None:
        segs = []
        for f in cand_files[:sample]:
            col = pq.read_table(f, columns=["entries"]).column("entries")
            for chunk in col.chunks:
                segs.append((chunk.offsets.to_numpy().copy(), chunk.values.to_numpy().copy()))
        _SAMPLE_CACHE = segs
    return _SAMPLE_CACHE


def expected_fill(p, cand_files, sample=20):
    """Expected accepted rows over the sample files, for budget scaling."""
    tot = 0; n = 0
    for offsets, vals in _load_sample(cand_files, sample):
        q = np.log1p(-np.minimum(p[vals], 1 - 1e-12))
        starts, ends = offsets[:-1], offsets[1:]
        ne = ends > starts
        seg = np.add.reduceat(q, starts[ne]) if ne.any() else np.array([])
        tot += (1 - np.exp(seg)).sum(); n += len(offsets) - 1
    return tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/weka/oe-training-default/chenhaoz/datacomp12b_supplement")
    ap.add_argument("--t", type=int, default=50000)
    ap.add_argument("--budget", type=float, default=0, help="target #samples (0 = natural deficit fill)")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    z = np.load(f"{a.out}/deficit_t{a.t}.npz")
    p = z["p"].astype(np.float64)
    files = sorted(glob.glob(f"{a.out}/cand/shard_*.parquet"))
    tag = f"selected_t{a.t}" + (f"_b{int(a.budget/1e6)}M" if a.budget else "")
    outdir = f"{a.out}/{tag}"; os.makedirs(outdir, exist_ok=True)
    if a.budget:
        # bisection on lambda so that expected accepted over ALL candidate files == budget
        ef, n = expected_fill(p, files); per_file = ef / min(len(files), 20)
        nat = per_file * len(files)
        print(f"natural fill ~{nat/1e6:.1f}M samples over {len(files)} shards; budget {a.budget/1e6:.0f}M")
        lo, hi = 0.0, 1.0 if a.budget < nat else 50.0
        for _ in range(40):
            lam = (lo + hi) / 2
            ef, _n = expected_fill(np.minimum(1, p * lam), files)
            if ef / min(len(files), 20) * len(files) < a.budget: lo = lam
            else: hi = lam
        p = np.minimum(1, p * lam); print(f"lambda={lam:.4f}")
    t0 = time.time()
    tot_in = tot_out = 0
    with Pool(a.procs, initializer=_init, initargs=(p,)) as pool:
        for i, (k, nin, nout) in enumerate(pool.imap_unordered(_one, [(f, outdir, a.seed * 100003 + i) for i, f in enumerate(files)])):
            tot_in += nin or 0; tot_out += nout
            if (i + 1) % 25 == 0 or i + 1 == len(files):
                print(f"  [{i+1}/{len(files)}] accepted {tot_out:,} of {tot_in:,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {tag}: {tot_out:,} selected samples -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
