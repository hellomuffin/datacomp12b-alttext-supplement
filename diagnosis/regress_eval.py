"""English run: are the ImageNet regressions optimizer/LR noise or bad data?

For each checkpoint triple around a regression:
  1. re-evaluate each checkpoint exactly as training does (build_zero_shot_classifier,
     OPENAI_IMAGENET_TEMPLATES, val transform rebuilt at the stage resolution) and check
     the logged top-1 is reproduced;
  2. evaluate the uniform weight average of the triple (SWA test: noise around a basin
     => the average beats every member);
  3. save per-image predictions for per-class churn analysis.
"""
import os, sys, json, time, argparse
import certifi
for _v in ("SSL_CERT_FILE","REQUESTS_CA_BUNDLE","CURL_CA_BUNDLE"): os.environ[_v]=certifi.where()
import torch, torch.nn.functional as F
from torchvision import datasets
sys.path.insert(0, "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/wt_run/src")
from open_clip import build_zero_shot_classifier, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip.transform import PreprocessCfg, merge_preprocess_dict, image_transform_v2
from open_clip_train.eval_model import load_eval_model, load_checkpoint_state

RUN=("/weka/oe-training-default/oe-encoder/checkpoints/ViT_SO400M_14/pretrain_v3/logs/"
     "v3_eng_lr0p003-wd0p05-lamb-wsd10ep-w3820-bs_131Kconst-bf16-cna_ct0p8af1p0g1p0-pgres98to378-k5-pmpr_l0p5-10B-32gB300")
VAL="/weka/oe-training-default/georges/datasets/imagenet1k/imagenet/val"
OUT="/weka/oe-mm-pretrain/oe-encoder/_orchestrator/regress"
ap=argparse.ArgumentParser(); ap.add_argument("--epochs", required=True); ap.add_argument("--res", type=int, required=True)
a=ap.parse_args(); eps=[int(x) for x in a.epochs.split(",")]
t0=time.time(); log=lambda s: print(f"[{time.time()-t0:6.0f}s] {s}", flush=True)

em=load_eval_model(f"{RUN}/checkpoints/epoch_{eps[0]}", image_size=a.res)
model=em.model.cuda().eval(); tok=em.tokenizer
log(f"model {em.model_name} built; rope={em.use_rope}")

# val transform exactly as main.py rebuilds it for a progressive stage
base=model.visual.preprocess_cfg
cfg=PreprocessCfg(**merge_preprocess_dict(base, {"size": a.res}))
pre=image_transform_v2(cfg, is_train=False)
cache=f"{OUT}/imagenet_val_r{a.res}.pt"
if os.path.exists(cache):
    X, Y = torch.load(cache); log(f"loaded cached val tensors {tuple(X.shape)}")
else:
    ds=datasets.ImageFolder(VAL, transform=pre)
    dl=torch.utils.data.DataLoader(ds, batch_size=500, num_workers=32)
    xs,ys=[],[]
    for i,(x,y) in enumerate(dl):
        xs.append(x.half()); ys.append(y)
        if i%10==0: log(f"  caching val r{a.res}: {sum(len(t) for t in ys)}/{len(ds)}")
    X=torch.cat(xs); Y=torch.cat(ys); torch.save((X,Y), cache); log(f"cached {tuple(X.shape)}")

@torch.no_grad()
def evaluate(tag):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clf=build_zero_shot_classifier(model, tokenizer=tok, classnames=IMAGENET_CLASSNAMES,
                                       templates=OPENAI_IMAGENET_TEMPLATES, num_classes_per_batch=10,
                                       device="cuda", use_tqdm=False)
    top5=[]
    for i in range(0, len(X), 500):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f=model.encode_image(X[i:i+500].cuda().float())
            if isinstance(f, dict):
                f=[v for k,v in f.items() if torch.is_tensor(v) and v.dim()==2][0]
            f=F.normalize(f.float(), dim=-1)
            logits=f @ clf.float()
        top5.append(logits.topk(5, dim=-1).indices.cpu())
    top5=torch.cat(top5)
    t1=(top5[:,0]==Y).float().mean().item(); t5=(top5==Y[:,None]).any(1).float().mean().item()
    torch.save({"top5":top5,"y":Y}, f"{OUT}/pred_{tag}_r{a.res}.pt")
    log(f"RESULT {tag:14s} res={a.res} top1={100*t1:.2f} top5={100*t5:.2f}")
    return t1

states=[]
for e in eps:
    sd=load_checkpoint_state(f"{RUN}/checkpoints/epoch_{e}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"epoch_{e} loaded (missing={len(missing)} unexpected={len(unexpected)})")
    states.append({k: v.float() for k,v in sd.items() if torch.is_tensor(v) and v.is_floating_point()})
    evaluate(f"epoch_{e}")

avg={k: sum(s[k] for s in states)/len(states) for k in states[0]}
model.load_state_dict(avg, strict=False)
evaluate("avg_" + "_".join(map(str,eps)))
log("DONE")
