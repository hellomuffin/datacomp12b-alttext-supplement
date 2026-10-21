"""Per-class zero-shot accuracy for the old-data and new-data checkpoints.

Mirrors the harness that produced eval_experiments/*.json
(/weka/oe-training-default/rustin/molmo-enc/eval.py): same DATASET_ROOT, same
clip_benchmark build_dataset + collate, same zero_shot_classifier and templates, same
336px eval size -- but keeps per-class correct/total so the regression can be correlated
with each class's keyword frequency in the two training pools.

Writes one JSON line per (run, dataset) to evid/perclass.jsonl; resumable.
"""
import argparse, json, os, sys
print("perclass_eval: starting", flush=True)
# NOTE: the model's tokenizer is an HF one (timm/ViT-B-16-SigLIP). Forcing offline mode
# makes get_tokenizer() fail, and on cluster nodes without egress the HF lookup hangs for
# over an hour -- so run this where HF is reachable (or with the tokenizer pre-cached).
import certifi
for _v in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ[_v] = certifi.where()
import ssl as _ssl
_ssl._create_default_https_context = lambda *a, **k: _ssl.create_default_context(cafile=certifi.where())
import torch
import torch.nn.functional as F
from torchvision import datasets as tv_datasets

REPO = "/weka/oe-training-default/jasonr/chenhao_prior/home/open_clip"
sys.path.insert(0, "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/wt_run/src")  # eval_model.py lives only in the running commit's worktree
sys.path.insert(0, f"{REPO}/CLIP_benchmark")
from open_clip.transform import PreprocessCfg, merge_preprocess_dict, image_transform_v2
from open_clip_train.eval_model import load_eval_model

DATASET_ROOT = "/weka/oe-training-default/rustin/clip_benchmark/{dataset}"
IMGFOLDER_FALLBACKS = {"cars": "cars_imgfolder", "sun397": "sun397_imgfolder"}
CKPT = "/weka/oe-training-default/oe-encoder/checkpoints"
RUNS = {
    "v0old_ep39": f"{CKPT}/v0fixed_so400m_dinov3_ms469K/epoch_39.pt",
    "v3_eng_ep48": (f"{CKPT}/ViT_SO400M_14/pretrain_v3/logs/v3_eng_lr0p003-wd0p05-lamb-wsd10ep-w3820-"
                    "bs_131Kconst-bf16-cna_ct0p8af1p0g1p0-pgres98to378-k5-pmpr_l0p5-10B-32gB300/checkpoints/epoch_48"),
    "v4_eng_latest": (f"{CKPT}/ViT_SO400M_14/pretrain_v3_eng/logs/v4_eng_lr1e-3-wd1e-4-adamw-cos-w3820-"
                      "bs_131Kconst-bf16-cna_ct0p8af1p0g1p0-pgres98to378-k3-pmpr_l0p5-10B-32gB300/checkpoints/epoch_latest"),
}
# v0old's checkpoint dir has no params.txt; the colleague's eval recorded this one
PARAMS_TXT = {"v0old_ep39": "/weka/oe-training-default/georges/checkpoints/ViT_B_16/v0/DataComp1B_MetaclipBased_mvit/logs2/"
                            "v0fixed_lr0p001-wd0p0001-opt_adamw-w20K-b10p9-b20p95-bs_16p4K-pamp_bf16-l1p0-"
                            "cna_ct0p8af1p0g1p0-pgres-pmpr_l0p5-ms469K-rope_dp/params.txt"}
DATASETS = ["flowers", "food101", "pets", "fgvc_aircraft", "cars"]
RES = 336
OUT = "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/evid/perclass.jsonl"


def build(ds, preprocess):
    from clip_benchmark.datasets.builder import build_dataset
    root = DATASET_ROOT.format(dataset=ds)
    try:
        return build_dataset(dataset_name=ds, root=root, transform=preprocess, split="test", download=False)
    except Exception as e:
        fallback = IMGFOLDER_FALLBACKS.get(ds)
        fp = os.path.join(os.path.dirname(root), fallback) if fallback else None
        if not (fp and os.path.isdir(fp)):
            raise
        print(f"  {ds}: build_dataset failed ({e}); ImageFolder fallback {fp}", flush=True)
        d = tv_datasets.ImageFolder(fp, transform=preprocess)
        cn_path = os.path.join(os.path.dirname(root), "custom_classnames.json")
        with open(cn_path) as cf:
            orig = json.load(cf).get(ds)
        if orig:
            d.classes = [orig[int(x)] for x in sorted(os.listdir(fp))]
        d.templates = ["a photo of a {c}.", "a bad photo of a {c}.", "a photo of the {c}."]
        return d


def _vec(out):
    if isinstance(out, dict):
        c = [v for v in out.values() if torch.is_tensor(v) and v.dim() == 2]
        assert len(c) == 1, sorted(out)
        return c[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    have = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            have.add((r["run"], r["dataset"]))
    for run in a.runs.split(","):
        todo = [d for d in a.datasets.split(",") if (run, d) not in have]
        if not todo:
            print(f"[{run}] all datasets done", flush=True)
            continue
        print(f"[{run}] loading checkpoint {RUNS[run]}", flush=True)
        em = load_eval_model(RUNS[run], params_txt=PARAMS_TXT.get(run), image_size=RES)
        model = em.model.cuda().eval()
        cfg = PreprocessCfg(**merge_preprocess_dict(model.visual.preprocess_cfg, {"size": RES}))
        pre = image_transform_v2(cfg, is_train=False)
        print(f"[{run}] model ready ({em.model_name})", flush=True)
        for ds in todo:
            from clip_benchmark.datasets.builder import get_dataset_collate_fn
            from clip_benchmark.metrics.zeroshot_classification import zero_shot_classifier
            d = build(ds, pre)
            print(f"[{run}] {ds}: {len(d)} images, {len(d.classes)} classes", flush=True)
            clf = zero_shot_classifier(model, em.tokenizer, d.classes, d.templates, "cuda")
            dl = torch.utils.data.DataLoader(d, batch_size=128, num_workers=8,
                                             collate_fn=get_dataset_collate_fn(ds))
            ncls = len(d.classes)
            correct = torch.zeros(ncls)
            total = torch.zeros(ncls)
            seen = 0
            with torch.no_grad():
                for images, target in dl:
                    with torch.amp.autocast("cuda"):
                        f = F.normalize(_vec(model.encode_image(images.cuda())).float(), dim=-1)
                    pred = (f @ clf.float()).argmax(-1).cpu()
                    for yi, pi in zip(target.cpu().tolist(), pred.tolist()):
                        total[yi] += 1
                        correct[yi] += (yi == pi)
                    seen += len(target)
                    if seen % 2560 < 128:
                        print(f"[{run}] {ds}: {seen}/{len(d)}", flush=True)
            top1 = 100 * correct.sum().item() / max(1.0, total.sum().item())
            with open(OUT, "a") as fh:
                fh.write(json.dumps({"run": run, "dataset": ds, "top1": round(top1, 2),
                                     "n": int(total.sum().item()),
                                     "per_class": {d.classes[i]: [int(correct[i]), int(total[i])]
                                                   for i in range(ncls)}}) + "\n")
            print(f"[{run}] {ds}: top1={top1:.2f}", flush=True)
        del model, em
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
