"""Merge the per-tar keyword counts into a calculated comparison of the two pools.

Outputs (evid/kwstats.json plus a printed report):
  * corpus stats per pool: samples, words/sample, empty and duplicate alt-text rates
  * per-class rate per million samples for every benchmark vocabulary, for alt-text,
    dense captions, and either, with the new/metaclip ratio per class
  * per-vocabulary aggregates: macro mean rate, median rate, how many classes are
    never mentioned, and how many fall below 1 and 10 per million
  * unigram distribution comparison: vocabulary sizes, coverage of the top-k,
    Jensen-Shannon divergence, and the tokens with the largest log-ratio each way
"""
import collections, glob, gzip, json, math, os, sys

ROOT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/kw"
OUT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/evid/kwstats.json"


def load(pool):
    st = collections.Counter()
    uni_a, uni_d = collections.Counter(), collections.Counter()
    cls_a, cls_d = collections.Counter(), collections.Counter()
    files = sorted(glob.glob(f"{ROOT}/{pool}/*.json.gz"))
    for i, f in enumerate(files):
        try:
            d = json.load(gzip.open(f, "rt"))
        except Exception as e:
            print(f"  skip unreadable {f}: {e}", flush=True)
            continue
        st.update(d["stats"])
        uni_a.update(d["uni_alt"])
        uni_d.update(d["uni_dense"])
        cls_a.update(d["cls_alt"])
        cls_d.update(d["cls_dense"])
        if (i + 1) % 50 == 0:
            print(f"  {pool}: merged {i+1}/{len(files)} shards", flush=True)
    return dict(n_shards=len(files), stats=st, uni_alt=uni_a, uni_dense=uni_d,
                cls_alt=cls_a, cls_dense=cls_d)


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits between two count distributions."""
    tp, tq = sum(p.values()), sum(q.values())
    keys = set(p) | set(q)
    d = 0.0
    for k in keys:
        pi, qi = p.get(k, 0) / tp, q.get(k, 0) / tq
        mi = 0.5 * (pi + qi)
        if pi:
            d += 0.5 * pi * math.log2(pi / mi)
        if qi:
            d += 0.5 * qi * math.log2(qi / mi)
    return d


def main():
    P = {pool: load(pool) for pool in ("new", "metaclip")}
    for pool in P:
        assert P[pool]["n_shards"], f"no counts for {pool}"
    res = {"corpus": {}, "vocab_summary": {}, "per_class": {}, "unigram": {}}

    for pool, d in P.items():
        s = d["stats"]
        n = s["n"]
        res["corpus"][pool] = {
            "shards": d["n_shards"], "samples": n,
            "alt_words_per_sample": round(s["alt_words"] / n, 2),
            "alt_chars_per_sample": round(s["alt_chars"] / n, 1),
            "dense_words_per_sample": round(s["dense_words"] / n, 2),
            "alt_empty_pct": round(100 * s["alt_empty"] / n, 3),
            "dense_empty_pct": round(100 * s["dense_empty"] / n, 3),
            "alt_with_digit_pct": round(100 * s["alt_has_digit"] / n, 2),
            "dup_alt_within_shard_pct": round(100 * s["dup_alt_captions"] / n, 2),
            "alt_unigram_vocab": len(d["uni_alt"]),
            "dense_unigram_vocab": len(d["uni_dense"]),
        }

    # ---- per class, rate per million samples ----
    vocabs = collections.defaultdict(set)
    for pool, d in P.items():
        for key in list(d["cls_alt"]) + list(d["cls_dense"]):
            v, c = key.split("::", 1)
            vocabs[v].add(c)
    # include classes with zero hits everywhere by reading the vocab lists back
    sys.path.insert(0, "/weka/oe-mm-pretrain/oe-encoder/_orchestrator")
    from kwcount import vocabs as all_vocabs
    V = all_vocabs()
    for v, classes in V.items():
        vocabs[v].update(classes)

    for v, classes in sorted(vocabs.items()):
        rows = {}
        for c in sorted(classes):
            key = f"{v}::{c}"
            row = {}
            for pool, d in P.items():
                n = d["stats"]["n"]
                a = 1e6 * d["cls_alt"].get(key, 0) / n
                de = 1e6 * d["cls_dense"].get(key, 0) / n
                row[pool] = {"alt_per_m": round(a, 2), "dense_per_m": round(de, 2)}
            for field in ("alt_per_m", "dense_per_m"):
                nw, mc = row["new"][field], row["metaclip"][field]
                row[f"ratio_{field}"] = round(nw / mc, 3) if mc > 0 else (None if nw == 0 else float("inf"))
            rows[c] = row
        res["per_class"][v] = rows

        def agg(pool, field):
            xs = [rows[c][pool][field] for c in rows]
            xs_sorted = sorted(xs)
            return {
                "macro_mean_per_m": round(sum(xs) / len(xs), 2),
                "median_per_m": round(xs_sorted[len(xs) // 2], 2),
                "classes_zero": sum(x == 0 for x in xs),
                "classes_below_1_per_m": sum(x < 1 for x in xs),
                "classes_below_10_per_m": sum(x < 10 for x in xs),
            }
        res["vocab_summary"][v] = {
            "n_classes": len(rows),
            "alt": {pool: agg(pool, "alt_per_m") for pool in P},
            "dense": {pool: agg(pool, "dense_per_m") for pool in P},
        }

    # ---- unigram distributions ----
    for field in ("uni_alt", "uni_dense"):
        a, b = P["new"][field], P["metaclip"][field]
        ta, tb = sum(a.values()), sum(b.values())
        res["unigram"][field] = {
            "js_divergence_bits": round(js_divergence(a, b), 4),
            "vocab_new": len(a), "vocab_metaclip": len(b),
            "tokens_new": ta, "tokens_metaclip": tb,
        }
        # tokens with the largest shift, restricted to reasonably frequent ones
        common = [t for t in set(a) | set(b)
                  if a.get(t, 0) + b.get(t, 0) >= 200 and len(t) > 2]
        lr = {}
        for t in common:
            pa = (a.get(t, 0) + 1) / ta
            pb = (b.get(t, 0) + 1) / tb
            lr[t] = math.log2(pa / pb)
        top_new = sorted(lr.items(), key=lambda x: -x[1])[:40]
        top_meta = sorted(lr.items(), key=lambda x: x[1])[:40]
        res["unigram"][field]["over_in_new"] = [
            (t, round(v, 2), a.get(t, 0), b.get(t, 0)) for t, v in top_new]
        res["unigram"][field]["over_in_metaclip"] = [
            (t, round(v, 2), a.get(t, 0), b.get(t, 0)) for t, v in top_meta]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    # ---- report ----
    print("\n=== corpus ===")
    for pool, c in res["corpus"].items():
        print(f"{pool:9s} shards={c['shards']:4d} samples={c['samples']:>9,} "
              f"alt_words/sample={c['alt_words_per_sample']:5.2f} dense_words={c['dense_words_per_sample']:6.2f} "
              f"alt_empty={c['alt_empty_pct']:.2f}% dup_alt={c['dup_alt_within_shard_pct']:.2f}% "
              f"alt_vocab={c['alt_unigram_vocab']:,}")
    print("\n=== per-vocabulary class-name rate (per million samples) ===")
    print(f"{'vocab':14s}{'n':>5s}  {'alt new':>9s}{'alt meta':>9s}{'ratio':>7s}   "
          f"{'dense new':>10s}{'dense meta':>11s}{'ratio':>7s}   {'new <1/M':>9s}{'meta <1/M':>10s}")
    for v, s in res["vocab_summary"].items():
        an, am = s["alt"]["new"]["macro_mean_per_m"], s["alt"]["metaclip"]["macro_mean_per_m"]
        dn, dm = s["dense"]["new"]["macro_mean_per_m"], s["dense"]["metaclip"]["macro_mean_per_m"]
        print(f"{v:14s}{s['n_classes']:5d}  {an:9.1f}{am:9.1f}{(an/am if am else 0):7.2f}   "
              f"{dn:10.1f}{dm:11.1f}{(dn/dm if dm else 0):7.2f}   "
              f"{s['alt']['new']['classes_below_1_per_m']:9d}{s['alt']['metaclip']['classes_below_1_per_m']:10d}")
    print("\n=== unigram distribution ===")
    for field, u in res["unigram"].items():
        print(f"{field}: JS={u['js_divergence_bits']:.4f} bits  vocab new={u['vocab_new']:,} "
              f"metaclip={u['vocab_metaclip']:,}")
        print("   over-represented in NEW:     " + ", ".join(t for t, *_ in u["over_in_new"][:15]))
        print("   over-represented in METACLIP:" + ", ".join(t for t, *_ in u["over_in_metaclip"][:15]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
