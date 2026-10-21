"""Statistics of a balanced selection: per-entry fill vs deficit, overfill, and the
resulting pool+supplement profile vs the t target. Also benchmark-class fills."""
import argparse, glob, json, os
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

META = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/reference_code/MetaCLIP/metaclip/metaclip1/metadata.json"
E = "/weka/oe-training-default/oe-encoder/eval_data/clip_benchmark"
CB = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/CLIP_benchmark/clip_benchmark/datasets/en_classnames.json"
N = 500_000


def _count(path):
    c = np.zeros(N, np.int64); n = 0; words = 0
    for chunk in pq.ParquetFile(path).iter_batches(batch_size=200_000, columns=["entries", "text"]):
        col = chunk.column("entries")
        vals = col.combine_chunks().values.to_numpy() if hasattr(col, "combine_chunks") else col.values.to_numpy()
        np.add.at(c, vals, 1)
        n += len(col)
        words += sum(len(t.split()) for t in chunk.column("text").to_pylist())
    return c, n, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/weka/oe-training-default/chenhaoz/datacomp12b_supplement")
    ap.add_argument("--t", type=int, default=50000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    tag = a.tag or f"selected_t{a.t}"
    files = sorted(glob.glob(f"{a.out}/{tag}/shard_*.parquet"))
    z = np.load(f"{a.out}/deficit_t{a.t}.npz")
    c_est, d, T, avail = z["c_est"], z["d"], z["T"], z["avail"]
    with Pool(a.procs) as p:
        res = p.map(_count, files)
    sel = sum(r[0] for r in res); n = sum(r[1] for r in res); words = sum(r[2] for r in res)
    size_gb = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"{tag}: {n:,} samples from {len(files)} shards, {size_gb:.1f} GB parquet, {words/max(1,n):.1f} words/caption")
    has = d > 0
    fill = np.where(has, np.minimum(sel, d) / np.maximum(d, 1), np.nan)
    print(f"entries with deficit: {has.sum():,}; fully filled (sel>=d): {(sel[has] >= d[has]).sum():,}; "
          f"median fill {np.nanmedian(fill):.2f}; mean fill {np.nanmean(fill):.2f}")
    over = sel - d
    print(f"overfill (sel-d, entries with deficit): total {over[has][over[has]>0].sum()/1e6:.1f}M hits over "
          f"{(over[has]>0).sum():,} entries; hits on entries with NO deficit: {sel[~has].sum()/1e6:.1f}M")
    new_c = c_est + sel
    print(f"after supplement: entries below t: {(new_c < a.t).sum():,} (before: {(c_est < a.t).sum():,}); "
          f"entries below t/2: {(new_c < a.t/2).sum():,} (before: {(c_est < a.t/2).sum():,})")
    meta = json.load(open(META)); idx = {e.lower(): i for i, e in enumerate(meta)}
    cn = json.load(open(CB)); cu = json.load(open(f"{E}/custom_classnames.json"))
    food = [l.strip().replace("_", " ") for l in open(f"{E}/food101/food-101/meta/classes.txt") if l.strip()]
    for v, classes in {"flowers": cn["flowers"], "pets": cn["pets"], "food101": food, "sun397": cu["sun397"]}.items():
        rows = [(cl, idx[cl.lower()]) for cl in classes if cl.lower() in idx]
        before = sum(c_est[j] for _, j in rows); add = sum(sel[j] for _, j in rows)
        print(f"  {v:8s}: {len(rows)} entry-classes; pool informative hits {before:,.0f} -> +{add:,} ({100*add/max(1,before):.0f}%)")
        for cl, j in sorted(rows, key=lambda r: -d[r[1]])[:4]:
            print(f"     {cl:26s} c~{c_est[j]:7,.0f} d={d[j]:6,.0f} avail~{avail[j]:7,.0f} selected={sel[j]:6,}")
    np.save(f"{a.out}/{tag}_entry_counts.npy", sel)


if __name__ == "__main__":
    main()
