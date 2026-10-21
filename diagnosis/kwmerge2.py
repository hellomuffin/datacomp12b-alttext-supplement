"""Merge the second-pass counts: Cars matched by make+model, and per-sentence dense rates.

Per-sentence rate is the number that matters for training: each positive slot gets either
the alt-text (probability ~0.5, giving the alt rate) or ONE sentence of the dense caption,
so a class's chance of being named in a sampled dense positive is the per-sentence rate.
"""
import collections, glob, gzip, json

ROOT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/kw2"
OUT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/evid/kwstats2.json"
VOCABS = ["cars_full", "cars_noyear", "cars_make_model", "flowers", "pets",
          "food101", "fgvc_aircraft", "sun397", "imagenet1k"]


def load(pool):
    st = collections.Counter()
    alt, dcap, dsent = collections.Counter(), collections.Counter(), collections.Counter()
    files = sorted(glob.glob(f"{ROOT}/{pool}/*.json.gz"))
    for f in files:
        d = json.load(gzip.open(f, "rt"))
        st.update(d["stats"])
        alt.update(d["cls_alt"])
        dcap.update(d["cls_dense_cap"])
        dsent.update(d["cls_dense_sent"])
    return dict(files=len(files), stats=st, alt=alt, dcap=dcap, dsent=dsent)


def main():
    P = {p: load(p) for p in ("new", "metaclip")}
    res = {"corpus": {}, "per_vocab": {}, "per_class": {}}
    for p, d in P.items():
        n = d["stats"]["n"]
        res["corpus"][p] = {"tars": d["files"], "samples": n,
                            "sentences_per_dense_caption": round(d["stats"]["n_sentences"] / n, 2)}
        print(f"{p:9s} tars={d['files']:3d} samples={n:>8,} "
              f"sentences/dense_caption={d['stats']['n_sentences']/n:.2f}", flush=True)

    print(f"\n{'vocab':18s}{'alt new':>9s}{'alt meta':>9s}{'ratio':>7s}"
          f"{'sent new':>10s}{'sent meta':>10s}{'ratio':>7s}"
          f"{'cap new':>9s}{'cap meta':>9s}{'ratio':>7s}")
    for v in VOCABS:
        agg = {}
        classes = set()
        for p, d in P.items():
            for src in ("alt", "dcap", "dsent"):
                for k in d[src]:
                    if k.startswith(v + "::"):
                        classes.add(k.split("::", 1)[1])
        if not classes:
            continue
        rows = {}
        for c in sorted(classes):
            key = f"{v}::{c}"
            row = {}
            for p, d in P.items():
                n = d["stats"]["n"]
                ns = max(1, d["stats"]["n_sentences"])
                row[p] = {"alt_per_m": round(1e6 * d["alt"].get(key, 0) / n, 2),
                          "dense_cap_per_m": round(1e6 * d["dcap"].get(key, 0) / n, 2),
                          "dense_sent_per_m_sentences": round(1e6 * d["dsent"].get(key, 0) / ns, 2)}
            rows[c] = row
        res["per_class"][v] = rows

        def macro(p, field):
            xs = [rows[c][p][field] for c in rows]
            return sum(xs) / len(xs)
        a_n, a_m = macro("new", "alt_per_m"), macro("metaclip", "alt_per_m")
        s_n, s_m = macro("new", "dense_sent_per_m_sentences"), macro("metaclip", "dense_sent_per_m_sentences")
        c_n, c_m = macro("new", "dense_cap_per_m"), macro("metaclip", "dense_cap_per_m")
        r = lambda x, y: (x / y if y else 0)
        res["per_vocab"][v] = {"n_classes_seen": len(rows),
                               "alt": [round(a_n, 2), round(a_m, 2), round(r(a_n, a_m), 3)],
                               "dense_sentence": [round(s_n, 2), round(s_m, 2), round(r(s_n, s_m), 3)],
                               "dense_caption": [round(c_n, 2), round(c_m, 2), round(r(c_n, c_m), 3)]}
        print(f"{v:18s}{a_n:9.1f}{a_m:9.1f}{r(a_n,a_m):7.2f}"
              f"{s_n:10.1f}{s_m:10.1f}{r(s_n,s_m):7.2f}"
              f"{c_n:9.1f}{c_m:9.1f}{r(c_n,c_m):7.2f}")

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
