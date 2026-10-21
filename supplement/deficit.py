"""Supplement step 3: per-entry deficit table from the single-pass scan (cand/shard_*.npz).

  c_j    informative alt-text hits of entry j in the CURRENT pool (counts_pool_inf, scaled to 1000 shards)
  s_j    informative hits among scanned UNUSED candidates (counts_cand_inf)
  a_j    estimated total availability = s_j * 1000 / n_shards_scanned
  T_j    target = min(t, c_j + a_j)          (t = balancing threshold, MetaCLIP semantics)
  d_j    deficit = max(0, T_j - c_j)
  p_j    acceptance prob for scanned candidates = min(1, d_j / s_j)
         (MetaCLIP rule with t replaced by the deficit; expected fill = d_j * scanned/1000)

Writes <out>/deficit_t{t}.npz and prints a summary plus a benchmark-class view.
"""
import argparse, glob, json
import numpy as np

META = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/reference_code/MetaCLIP/metaclip/metaclip1/metadata.json"
E = "/weka/oe-training-default/oe-encoder/eval_data/clip_benchmark"
CB = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip/CLIP_benchmark/clip_benchmark/datasets/en_classnames.json"
STATS = ["n_rows", "n_english", "n_pool", "n_pool_inf", "n_pool_dup", "n_pool_placeholder", "n_pool_short",
         "n_pool_urlish", "n_unused", "n_dup", "n_placeholder", "n_short", "n_urlish", "n_kept"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/weka/oe-training-default/chenhaoz/datacomp12b_supplement")
    ap.add_argument("--t", type=int, default=50000)
    ap.add_argument("--universe", type=int, default=987, help="shards that exist in the metadata dump (995-999 and 7 others are absent)")
    a = ap.parse_args()
    meta = json.load(open(META))
    files = sorted(glob.glob(f"{a.out}/cand/shard_*.npz"))
    assert files, "no scanned shards"
    c = np.zeros(len(meta), np.int64); c_all = np.zeros(len(meta), np.int64); s = np.zeros(len(meta), np.int64)
    st = {k: 0 for k in STATS}
    for f in files:
        z = np.load(f)
        c += z["counts_pool_inf"]; c_all += z["counts_pool_all"]; s += z["counts_cand_inf"]
        for k in STATS:
            st[k] += int(z[k])
    n_sc = len(files); scale = a.universe / n_sc
    c_est = c * scale; avail = s * scale
    T = np.minimum(a.t, c_est + avail)
    d = np.maximum(0, T - c_est)
    p = np.where(s > 0, np.minimum(1.0, d / np.maximum(s, 1)), 0.0)
    np.savez(f"{a.out}/deficit_t{a.t}.npz", c=c, c_all=c_all, c_est=c_est, s=s, avail=avail, T=T, d=d, p=p,
             n_shards_scanned=n_sc, t=a.t, **{k: v for k, v in st.items()})

    print(f"scanned {n_sc}/{a.universe} existing shards (scale x{scale:.3f})")
    print(f"rows {st['n_rows']:,} | english {st['n_english']:,} | pool {st['n_pool']:,} (~{st['n_pool']*scale/1e9:.2f}B) "
          f"| unused {st['n_unused']:,} (~{st['n_unused']*scale/1e9:.2f}B) | kept candidates {st['n_kept']:,} (~{st['n_kept']*scale/1e9:.2f}B)")
    np_ = max(1, st["n_pool"]); nu = max(1, st["n_unused"])
    print(f"POOL alt-text:   informative {100*st['n_pool_inf']/np_:.1f}%  placeholder {100*st['n_pool_placeholder']/np_:.2f}%  "
          f"short {100*st['n_pool_short']/np_:.1f}%  urlish {100*st['n_pool_urlish']/np_:.2f}%  dup {100*st['n_pool_dup']/np_:.2f}%")
    print(f"UNUSED alt-text: informative {100*st['n_kept']/nu:.1f}%  placeholder {100*st['n_placeholder']/nu:.2f}%  "
          f"short {100*st['n_short']/nu:.1f}%  urlish {100*st['n_urlish']/nu:.2f}%  dup {100*st['n_dup']/nu:.2f}%")
    has = d > 0
    print(f"\nt={a.t}: entries present in pool {(c_all>0).sum():,}; entries with deficit {has.sum():,}; "
          f"total deficit {d.sum()/1e6:.1f}M entry-hits")
    hits_per = s.sum() / max(1, st["n_kept"])
    exp_fill = (p * s).sum() * scale
    print(f"mean entries per candidate {hits_per:.2f}; expected accepted entry-hits (all shards) {exp_fill/1e6:.1f}M "
          f"-> ~{exp_fill/hits_per/1e6:.0f}M samples (overlap-corrected estimate comes from balance_select)")
    frac = np.where(T > 0, d / np.maximum(T, 1), 0)
    print("\ndeficit as fraction of target (entries with deficit):")
    for lo, hi in [(0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        sel = has & (frac >= lo) & (frac < hi)
        print(f"  {lo:.1f}-{hi:.1f}: {sel.sum():7,} entries, deficit {d[sel].sum()/1e6:8.1f}M")

    cn = json.load(open(CB)); cu = json.load(open(f"{E}/custom_classnames.json"))
    food = [l.strip().replace("_", " ") for l in open(f"{E}/food101/food-101/meta/classes.txt") if l.strip()]
    vocabs = {"flowers": cn["flowers"], "pets": cn["pets"], "food101": food, "sun397": cu["sun397"]}
    idx = {e.lower(): i for i, e in enumerate(meta)}
    print("\nbenchmark classes that are MetaCLIP entries (pool informative hits est. -> target, deficit):")
    for v, classes in vocabs.items():
        rows = [(cl, idx[cl.lower()]) for cl in classes if cl.lower() in idx]
        tot_d = sum(d[j] for _, j in rows); tot_c = sum(c_est[j] for _, j in rows)
        print(f"  {v}: {len(rows)}/{len(classes)} classes are entries; pool inf hits ~{tot_c:,.0f}; total deficit {tot_d:,.0f}")
        for cl, j in sorted(rows, key=lambda r: -d[r[1]])[:5]:
            print(f"     {cl:28s} c~{c_est[j]:8,.0f} avail~{avail[j]:9,.0f} T={T[j]:7,.0f} d={d[j]:7,.0f} p={p[j]:.2f}")


if __name__ == "__main__":
    main()
