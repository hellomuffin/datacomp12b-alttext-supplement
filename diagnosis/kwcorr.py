"""Does the per-class accuracy regression follow the per-class keyword shift?

Joins evid/kwstats.json (per-class keyword rate per million in each pool) with
evid/perclass.jsonl (per-class accuracy for the old-data and new-data checkpoints) and
reports, per benchmark and pooled:
  * Spearman and Pearson correlation between the new/metaclip keyword-rate ratio and the
    per-class accuracy delta (new run - old run)
  * accuracy delta averaged over classes binned by keyword-rate ratio
  * the classes that lost the most accuracy, with their keyword rates in both pools
A positive correlation means classes whose keywords got rarer are the ones that regressed.
"""
import json, math, collections

EV = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/evid"
NEW_RUN = "v3_eng_ep48"
OLD_RUN = "v0old_ep39"
VOCAB_OF = {"flowers": "flowers", "food101": "food101", "pets": "pets",
            "fgvc_aircraft": "fgvc_aircraft", "cars": "cars"}


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(x, y):
    return pearson(rank(x), rank(y))


def main():
    kw = json.load(open(f"{EV}/kwstats.json"))
    acc = {}
    for line in open(f"{EV}/perclass.jsonl"):
        r = json.loads(line)
        acc[(r["run"], r["dataset"])] = r

    pooled = {"ratio": [], "delta": [], "logratio": [], "newrate": []}
    print(f"{'dataset':14s}{'n':>4s}{'spearman':>10s}{'pearson':>9s}   "
          f"{'mean delta':>11s}{'median kw ratio':>16s}")
    per_dataset = {}
    for ds, vocab in VOCAB_OF.items():
        if (NEW_RUN, ds) not in acc or (OLD_RUN, ds) not in acc:
            print(f"{ds:14s}  (missing accuracy rows)")
            continue
        new_pc = acc[(NEW_RUN, ds)]["per_class"]
        old_pc = acc[(OLD_RUN, ds)]["per_class"]
        rows = kw["per_class"][vocab]
        xs, ys, names = [], [], []
        for cls, (c_new, t) in new_pc.items():
            if cls not in old_pc or cls not in rows or t == 0:
                continue
            c_old = old_pc[cls][0]
            delta = 100.0 * (c_new - c_old) / t
            n_rate = rows[cls]["new"]["alt_per_m"] + rows[cls]["new"]["dense_per_m"]
            m_rate = rows[cls]["metaclip"]["alt_per_m"] + rows[cls]["metaclip"]["dense_per_m"]
            if m_rate <= 0 and n_rate <= 0:
                continue
            ratio = (n_rate + 0.1) / (m_rate + 0.1)
            xs.append(ratio)
            ys.append(delta)
            names.append((cls, delta, n_rate, m_rate, ratio))
            pooled["ratio"].append(ratio)
            pooled["delta"].append(delta)
            pooled["logratio"].append(math.log2(ratio))
            pooled["newrate"].append(n_rate)
        if len(xs) < 3:
            continue
        sp, pe = spearman(xs, ys), pearson([math.log2(v) for v in xs], ys)
        med = sorted(xs)[len(xs) // 2]
        print(f"{ds:14s}{len(xs):4d}{sp:10.3f}{pe:9.3f}   {sum(ys)/len(ys):11.2f}{med:16.2f}")
        per_dataset[ds] = {"n_classes": len(xs), "spearman_ratio_vs_delta": round(sp, 3),
                           "pearson_log2ratio_vs_delta": round(pe, 3),
                           "mean_delta": round(sum(ys) / len(ys), 2),
                           "median_kw_ratio": round(med, 3),
                           "worst_classes": [
                               {"class": c, "delta": round(d, 1), "new_per_m": round(nr, 2),
                                "metaclip_per_m": round(mr, 2), "ratio": round(rt, 3)}
                               for c, d, nr, mr, rt in sorted(names, key=lambda z: z[1])[:10]]}

    print("\n=== pooled over all five benchmarks ===")
    sp = spearman(pooled["ratio"], pooled["delta"])
    pe = pearson(pooled["logratio"], pooled["delta"])
    print(f"classes={len(pooled['ratio'])}  spearman(kw ratio, acc delta)={sp:.3f}  "
          f"pearson(log2 kw ratio, acc delta)={pe:.3f}")

    bins = [(0, 0.5), (0.5, 0.8), (0.8, 1.25), (1.25, 2.0), (2.0, 1e9)]
    print(f"\n{'kw ratio bin':16s}{'classes':>8s}{'mean acc delta':>16s}")
    for lo, hi in bins:
        sel = [d for r, d in zip(pooled["ratio"], pooled["delta"]) if lo <= r < hi]
        if sel:
            lbl = f"{lo:g}-{hi:g}" if hi < 1e9 else f">{lo:g}"
            print(f"{lbl:16s}{len(sel):8d}{sum(sel)/len(sel):16.2f}")

    out = {"per_dataset": per_dataset,
           "pooled": {"n_classes": len(pooled["ratio"]),
                      "spearman_ratio_vs_delta": round(sp, 3),
                      "pearson_log2ratio_vs_delta": round(pe, 3)},
           "runs": {"new": NEW_RUN, "old": OLD_RUN}}
    json.dump(out, open(f"{EV}/kwcorr.json", "w"), indent=1)
    print(f"\nwrote {EV}/kwcorr.json")


if __name__ == "__main__":
    main()
