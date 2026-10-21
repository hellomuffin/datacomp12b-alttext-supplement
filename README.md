# Supplementing the English pretraining pool: diagnosis, method, results

*2026-10-21 — oe-encoder data team.*

This page documents an investigation into a fine-grained-classification regression in our
SO400M pretraining runs on the new curated English pool, the measurements that located its
cause in the **alt-text** of the pool, and a pipeline that selects a balanced supplement from
the unused DataComp-12B English data. Everything below is computed, not estimated; the
scripts live in this directory and all intermediate artifacts are on weka
(`/weka/oe-training-default/chenhaoz/datacomp12b_supplement/`).

---

## Repository layout

| path | contents |
|---|---|
| `supplement/` | the supplement pipeline (scan, deficit, balanced selection, stats, launcher) |
| `diagnosis/` | the measurement scripts behind section 2: keyword-distribution counting (`kwcount*.py`, `kwmerge*.py`), per-class accuracy with the official harness (`perclass_eval.py`), correlation (`kwcorr.py`), and the checkpoint tests attributing ImageNet jitter to constant-LR optimizer noise (`regress_eval.py`, `evidence_suite.py`) |
| `results/` | computed tables: `kwstats*.json` (per-class rates in both pools), `deficit_t*_final.txt`, `select_t50000_stats.txt`, `kwcorr.json` |

Paths inside the scripts point at AI2 weka/GCS locations and are kept verbatim so every number
on this page can be traced to its inputs.

---

## 1. The problem

Two pretraining runs on the new pool (`v3_eng`, `v4_eng`) match or beat the previous-generation
runs on ImageNet and its variants, but lose heavily on fine-grained benchmarks.
All numbers below come from one harness (CLIP_benchmark 1.6.2, same commit, 336 px).

| run | training data | optimizer | IN-1k | IN-v2 | IN-A | IN-R | Food101 | Flowers | Pets | Aircraft | Cars |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0old ep39 | metaclip_1b | AdamW 1e-3 | 79.23 | 72.85 | 80.20 | 92.43 | **93.43** | **84.09** | **93.19** | 46.56 | 89.27 |
| pretrain_131k ep37 | metaclip_1b | AdamW 1e-3 | 79.60 | 73.59 | 81.20 | 93.11 | **93.72** | **83.57** | **93.24** | 49.53 | 91.92 |
| v4_eng ep29/50 | new pool | AdamW 1e-3 | 77.15 | 70.77 | 72.97 | 91.41 | 80.67 | 72.13 | 88.99 | 27.30 | 89.22 |
| v3_eng ep48/50 | new pool | Lamb 3e-3, k=5 | 80.00 | 74.02 | 82.57 | 94.43 | 81.67 | 50.41 | 87.38 | 40.83 | 92.66 |

Three things separate data from recipe:

* The good runs use the **same caption sampler** as the bad ones
  (`sample_k_positives_from_original_and_dense`, k=5, `original_caption_prob=0.5`); only
  `train_data` differs.
* Both new-pool runs regress on Food101/Pets regardless of optimizer, so AdamW does not fix it.
* The existing 1B-sample data ablations (`dabl_1a` = metaclip vs `dabl_1b` = new pool, same
  recipe) already show the same sign at epoch 15: Food101 75.12→71.66, Aircraft 18.90→15.99.

Flowers is a special case: on the *same* new data, AdamW/k=3 (v4) gets 72 and Lamb-3e-3/k=5 (v3)
gets 50, and per-class Flowers loss is uncorrelated with per-class keyword shift
(Spearman −0.09 over 95 classes). That part is recipe, not data, and is out of scope here.

---

## 2. Locating the cause: keyword distributions of the two pools

We counted, over **4.85 M samples of the new pool and 4.58 M of metaclip_1b** (887 shards),
how often each benchmark class name appears in the alt-text and in the dense caption
(exact phrase, MetaCLIP spacing; Cars matched by make+model because its names embed model years).

**Class-name rate per million samples**

| vocabulary | alt new | alt metaclip | **alt ratio** | dense new | dense metaclip | dense ratio | per-sentence ratio |
|---|---|---|---|---|---|---|---|
| cars (make+model) | 56.5 | 65.1 | **0.87** | 71.0 | 76.5 | 0.93 | 0.92 |
| flowers | 46.7 | 59.8 | **0.78** | 117.7 | 120.7 | 0.98 | 0.97 |
| pets | 28.4 | 48.3 | **0.59** | 84.3 | 94.9 | 0.89 | 0.88 |
| food101 | 45.0 | 66.2 | **0.68** | 131.1 | 122.9 | 1.07 | 0.98 |
| fgvc_aircraft | 6.0 | 10.5 | **0.57** | 11.6 | 14.8 | 0.78 | 0.76 |
| sun397 | 255.0 | 375.0 | **0.68** | 1802 | 1774 | 1.02 | 1.01 |
| imagenet1k | 106.0 | 142.8 | **0.74** | 893 | 912 | 0.98 | 0.99 |

Two facts follow directly:

1. **The deficit is in the human alt-text, not in the images.** The dense captions — produced by
   the same captioner on both pools — mention these classes at parity (0.9–1.1×), so the image
   content is there. The alt-text channel is 0.57–0.87× and the ordering matches the regression:
   Cars (0.87) did not regress, Pets (0.59) and Aircraft (0.57) regressed most.
2. **Dense captions cannot compensate.** Training samples *one sentence* of the ~11-sentence dense
   caption per positive slot; the per-sentence rates are at parity, so dense text supplies the
   same exposure to both runs while alt-text supplies less.

Why is the alt-text worse? Corpus statistics over the same samples:

| | new pool | metaclip_1b |
|---|---|---|
| alt-text words / sample | 9.1 | 13.1 |
| ≤2-word alt-text | **22.1 %** | 6.3 % |
| ≤4-word alt-text | **38.1 %** | 18.2 % |
| placeholder strings ("Untitled photo", "image", "Thumbnail", …) | **2.9 %** | 0.03 % |
| duplicate alt-text (within shard) | **4.9 %** | 0.2 % |
| JS divergence of alt-text unigrams | 0.042 bits | — |
| JS divergence of dense unigrams | 0.0034 bits | — |

The pipeline that built the new pool balanced MetaCLIP entries on **dense captions**
(`pipeline_v3/phase1_dense_match.py` → `phase3_select.py`) and never looked at alt-text, so
junk alt-text passed through at its natural DataComp-12B rate. metaclip_1b descends from
DataComp-1B, which is CLIP-score filtered and therefore has cleaner alt-text.

Per-class accuracy was re-measured with the colleague's exact harness to make sure the numbers
compare (v0old Flowers 84.08 vs logged 84.09; v3 Flowers 50.41 vs 50.41; Pets 93.19/87.46 vs
93.19/87.38).

---

## 3. Method: a balanced supplement selected on alt-text

**Framing.** Since the images for these concepts are already in the pool and only their
alt-text is missing, the supplement is really *buying captions*: we select unused samples by
their alt-text alone, which is free to scan (metadata only, no image download needed to score).

**Objective.** We follow MetaCLIP's balancing rule rather than inventing a new one. Zero-shot
accuracy on a concept grows roughly log-linearly with its pretraining count; with log utility
per concept and a fixed budget, maximizing total utility equalizes counts up to availability
("water-filling"), which is exactly MetaCLIP's *keep with probability min(1, t / count)*. The
pool that works (metaclip_1b) was built this way, and our own pool was built this way on dense
captions. We therefore:

1. **Count the current pool's exposure in the right channel.** For every MetaCLIP entry j,
   `c_j` = number of pool samples whose *informative* alt-text contains the entry.
2. **Define the deficit.** `T_j = min(t, c_j + a_j)` where `a_j` is how many unused informative
   samples mention j; `d_j = max(0, T_j − c_j)`.
3. **Accept candidates with the MetaCLIP rule, with the deficit in place of t.**
   `p_j = min(1, d_j / a_j)`; a candidate is kept if any of its matched entries passes its
   Bernoulli draw. This is embarrassingly parallel and reproduces, in expectation, each
   entry's deficit; an optional budget scales all `p_j` by a common λ found by bisection.

**Design choices and why**

| choice | reason |
|---|---|
| vocabulary = MetaCLIP's 500 K entries, **never benchmark class names** | selecting on benchmark names would curate on the test set; Flowers/Pets/Food stay held-out |
| "informative" = ≥4 word tokens, not a placeholder, not URL-like, deduplicated | these are exactly the failure classes measured in §2 (22 % ≤2 words, 2.9 % placeholders, 5 % dups) |
| threshold t = 50 000 (also 100 000) | the pool's own balancing threshold family; two tiers reported |
| counts of the current pool measured on its *actual* uid set | the packed pool is the dense-caption selection (`dense_metaclip_v1/selected/t105521`, 2.32 B), not the older alt-text selections; using the wrong uid set mis-states both exclusion and deficit |
| single pass over the metadata | pool counting, candidate filtering and tagging share one read of the 12.8 B-row dump |

### 3.1 The selection algorithm, step by step

**Objects.** *Entries* are MetaCLIP's 500 K metadata strings (WordNet synsets, Wikipedia titles,
common n-grams — e.g. `shiba inu`, `dog`, `kitchenette`, `fried rice`); never benchmark class names.
A caption *matches* entry *j* if the entry appears in it as a whole phrase; a typical informative
caption matches ~4 entries (`"cute shiba inu dog sleeping on a sofa"` → {`shiba inu`, `dog`,
`sleeping`, `sofa`}). *Informative* alt-text = ≥4 word tokens, not a placeholder, not URL-like,
not a duplicate; only such captions are counted anywhere below.

**Step 1 — count what the pool already has.** For every entry *j*: `c_j` = pool samples whose
informative alt-text matches *j*; `a_j` = unused candidates (informative alt-text) matching *j*,
i.e. how many more could be added.

**Step 2 — how many more each entry needs.** With threshold *t* (50 k here):
target `T_j = min(t, c_j + a_j)` ("bring *j* up to *t*, or to everything that exists"), deficit
`d_j = T_j − c_j ≥ 0`. Head entries (`dog`, `photo`) have `c_j ≫ t` so `d_j = 0`.

**Step 3 — deficits become per-entry acceptance probabilities.** `p_j = min(1, d_j / a_j)`.
Taking each of the `a_j` candidates that mention *j* with probability `p_j` adds `p_j·a_j = d_j`
of them in expectation — exactly the deficit. If `a_j ≤ d_j`, `p_j = 1`: take every one.

**Step 4 — the per-candidate rule (MetaCLIP's balancing rule).**

```
for each candidate x with matched entries E(x):
    accept x  if  ANY entry j in E(x) passes its own coin flip  (u_j < p_j)
```

A sample is kept if at least one concept it names is still needed. No shared state, so every
shard runs independently.

**Worked example (real numbers, t = 50 k).**

| entry | pool `c_j` | available `a_j` | target `T_j` | deficit `d_j` | `p_j` |
|---|---|---|---|---|---|
| `shiba inu` | 23,670 | 36,345 | 50,000 | 26,330 | 0.72 |
| `beagle` | 17,772 | 22,913 | 40,685 | 22,913 | **1.00** (take all) |
| `dog` | millions | — | 50,000 | 0 | 0 |
| `sofa` | > 50 k | — | 50,000 | 0 | 0 |

* `"cute shiba inu dog sleeping on a sofa"` → {shiba inu 0.72, dog 0, sleeping 0, sofa 0} → accepted with probability 0.72.
* `"beagle puppy at the park"` → {beagle 1.0, puppy 0, park 0} → always accepted.
* `"dog sleeping on a sofa"` → all entries saturated → never accepted.

The rule pulls in exactly the samples that carry under-represented concepts and ignores samples
that only repeat what the pool already has.

**Why "ANY", and its one side effect.** Tail concepts almost always co-occur with head concepts
in the same caption; requiring *all* entries to pass would let head neighbours veto every tail
sample. "Any" guarantees the tail gets filled. The side effect: the accepted `shiba inu` sample
also adds a hit to `dog`, `sleeping`, `sofa`; over 358 M samples that is the 2.75 B "overfill"
hits on entries that needed nothing, and why the natural selection is 358 M rather than the
~134 M obtained by dividing total deficit by entries-per-caption. The pool itself was built with
the same rule. A greedy set-cover variant (accept only if some entry's *remaining* deficit is
> 0, then decrement) gives the same tail fill at roughly half the size.

**The budget knob.** `--budget B` multiplies every `p_j` by one scalar λ, found by bisection so the
expected accepted count equals B (λ = 0.365 for 150 M). It scales tail and head alike; to shrink
while protecting the tail, prefer the set-cover variant.

**Why this is principled.** The rule *is* MetaCLIP's `min(1, t/count)`, with *t* replaced by the
remaining deficit so it adds to an existing pool instead of re-sampling from scratch; equalizing
counts up to availability maximizes total concept utility under the empirically log-linear
count→zero-shot-accuracy relation (water-filling); and the selection never sees benchmark class
names, so the benchmark gains in §5 are a held-out consequence of balancing 500 K entries.

**Matching.** Aho–Corasick over space-padded entries with MetaCLIP's `spacing()`, identical to
`metaclip1/substr_matching_ahocorasick.py`; 111 k captions / s / process.

---

## 4. Implementation

```
supplement/
  quality.py          alt-text filter + MetaCLIP matcher (shared)
  cand_scan.py        one pass per GCS metadata shard: pool counts + candidate tagging  (cluster, 8 procs)
  deficit.py          per-entry deficit table -> deficit_t{T}.npz
  balance_select.py   MetaCLIP acceptance with p_j = d_j/a_j, optional --budget       (local, 48 procs)
  select_stats.py     fill vs deficit, benchmark-class view
  submit_cand_scan.sh launcher: N jobs round-robin over shards, --reverse second wave
```

Inputs: `gs://datacomp1b-metadata-dump/datacomp12B_metadata/1000_shards` (uid, url, text),
`datacomp12b_english_uids_v2/shard_k.pkl` (English filter), `dense_metaclip_v1/selected/t105521/shard_k.pkl`
(pool membership, compound keys → uid).

Scale and wall-clock: 995 existing shards, **12.80 B rows read**; 180 forward + 180 reverse
CPU jobs (8 processes each) plus a 40-job cleanup wave, ≈ 1 h of scanning; selection over the
resulting ~300 GB of candidate parquet takes ~2 min on one 242-core node. Every stage is
resumable per shard.

Things that bit us (recorded in `problem.md`): the container CA bundle breaks GCS TLS
(fixed with `certifi`); the model's HF tokenizer lookup hangs for over an hour on nodes without
egress; and a helper named `select.py` shadowed the stdlib `select` module in this conda env
(it is a shared library resolved via `sys.path`), killing every later numpy import until renamed.

---

## 5. Results

**Universe (complete scan, 995 shards)**

| | samples |
|---|---|
| DataComp-12B English | 6.29 B |
| current pool (dense-balanced selection) | 2.32 B |
| unused English | 3.98 B |
| unused with informative alt-text (deduped) | **2.62 B** |

**Alt-text quality, measured on every pool row seen (2.32 B) and every unused row (3.98 B)**

| | pool | unused |
|---|---|---|
| informative | 72.5 % | 65.8 % |
| placeholder | 0.71 % | 4.39 % |
| <4 word tokens | 17.0 % | 27.9 % |
| duplicate caption | **9.27 %** | 1.47 % |

**Deficit** (against the pool's informative alt-text counts)

| t | entries below target | total deficit (entry-hits) |
|---|---|---|
| 50 000 | 392 638 of 405 712 present | 541 M |
| 100 000 | 397 185 | 817 M |

Most of the deficit sits in entries that are 10–90 % short of target (377 k entries); only 14 k
entries are essentially absent from the pool.

**Selections** (uid, url, text, entries; parquet per shard)

| selection | samples | note |
|---|---|---|
| `selected_t50000/` | **357.8 M** | natural fill at t = 50 k; 52 GB; 15.4 words / caption |
| `selected_t100000/` | **491.5 M** | natural fill at t = 100 k |
| `selected_t50000_b150M/` | **162.5 M** | t = 50 k, all p scaled by λ = 0.365 |

**Fill quality (t = 50 k):** of 391 932 entries with a deficit, the median entry receives 100 % of
it (mean 99 %); 147 855 are completely filled. For the benchmark classes that happen to be
MetaCLIP entries, informative alt-text exposure rises as follows — with no benchmark name in
the selection vocabulary:

| benchmark | entry-classes | pool informative hits | added | gain |
|---|---|---|---|---|
| Pets | 27 / 37 | 603 943 | +360 551 | **+60 %** |
| Food101 | 55 / 101 | 2 736 338 | +1 026 481 | **+38 %** |
| SUN397 | 196 / 397 | 61 376 506 | +22 935 288 | +37 % |
| Flowers | 61 / 102 | 4 259 938 | +881 551 | +21 % |

Tail classes get essentially everything that exists: Shiba Inu 23.7 k → +28.8 k, Beagle
17.8 k → +22.9 k, water lily 16.6 k → +19.4 k, oysters 24.2 k → +22.0 k (availability-capped).

**Why the supplement is larger than the summed deficit.** Σ d_j / (entries per candidate)
≈ 134 M, but the MetaCLIP rule keeps a sample if *any* of its ~4 entries passes, so head entries
that co-occur with tail entries are over-filled (2.75 B hits land on entries with no deficit).
This is inherent to the rule and is how the existing pool was built; a deficit-prioritized
set-cover variant would give the same tail coverage at roughly half the size.

---

## 6. Limitations and next steps

* **Selection is on text only.** Image quality/aesthetics of the supplement are DataComp-raw;
  the existing pool had the same property.
* **Uniform budget scaling under-fills tail entries** by the same factor as head ones; if the
  ~150 M tier is chosen, prefer the deficit-prioritized variant.
* **The existing pool's text channel should be fixed regardless:** 27 % of its alt-text is
  placeholder/short/duplicate. Treating such alt-text as missing in the positive sampler
  (fall back to a dense sentence) and making `original_caption_prob` per-sample by
  informativeness are free and complementary.
* **Validation plan** (1 B samples each, identical recipe, `dabl` infra): control; control +
  text fixes; control + supplement; both. Pass criteria: alt-text class-name ratios return to
  ≈1.0 under the §2 measurement, and Pets/Food101/Aircraft recover toward the metaclip runs.
* **To materialize:** `img2dataset` on `uid,url` of the chosen selection → dense captioner →
  caption side-cars → pack as `english_packed/0199+` → append to a v3 English manifest. The
  loader needs no change.

---

## 7. Reproduce

```bash
# 1-3  scan (cluster, needs the GCS secret in the job env)
bash supplement/submit_cand_scan.sh 180 0 cv        # forward wave
bash supplement/submit_cand_scan.sh 180 0 cr 1      # reverse wave
# 3    deficit table
python supplement/deficit.py --t 50000 --universe 995
# 4    selection (+ optional budget)
python supplement/balance_select.py --t 50000 --procs 48
python supplement/balance_select.py --t 50000 --budget 150e6 --procs 48
python supplement/select_stats.py --t 50000
```

Artifacts: `/weka/oe-training-default/chenhaoz/datacomp12b_supplement/{cand,deficit_t*.npz,selected_t*,README.md}`.
Keyword-distribution measurements of §2: `/weka/oe-mm-pretrain/oe-encoder/_orchestrator/evid/{kwstats.json,kwstats2.json,perclass.jsonl}`.
