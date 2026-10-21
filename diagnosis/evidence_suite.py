"""English run, res-154 stage (checkpoints epoch_19..epoch_27): concrete tests of
'LR/optimizer noise' vs 'bad data' for the ImageNet dips at eval 23 and 27.

T1 control averaging : does averaging help equally in windows WITHOUT a dip?
T2 interpolation     : is the path epoch_22 -> epoch_23 barrier-free (same basin)?
T3 weight geometry   : do consecutive updates oscillate (negative cosine) and does
                       distance grow sub-linearly with epoch gap?
T4 training-data fit : on a fixed held-out set of 8192 English training samples
                       (identical tensors for every checkpoint), does SigLIP loss /
                       retrieval also get worse at the dip? Bad data that damages the
                       model should hurt this too; pure noise on a 1-dataset metric need not.
"""
import os, sys, json, time, random
import certifi
for _v in ("SSL_CERT_FILE","REQUESTS_CA_BUNDLE","CURL_CA_BUNDLE"): os.environ[_v]=certifi.where()
import ssl as _ssl
_ssl._create_default_https_context = lambda *a, **k: _ssl.create_default_context(cafile=certifi.where())
import numpy as np, torch, torch.nn.functional as F
from torchvision import transforms as T
sys.path.insert(0, "/weka/oe-mm-pretrain/oe-encoder/_orchestrator/wt_run/src")
import open_clip_train.data as D
from open_clip_train.params import parse_args
from open_clip import build_zero_shot_classifier, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip_train.eval_model import load_eval_model, load_checkpoint_state

RUN=("/weka/oe-training-default/oe-encoder/checkpoints/ViT_SO400M_14/pretrain_v3/logs/"
     "v3_eng_lr0p003-wd0p05-lamb-wsd10ep-w3820-bs_131Kconst-bf16-cna_ct0p8af1p0g1p0-pgres98to378-k5-pmpr_l0p5-10B-32gB300")
OUT="/weka/oe-mm-pretrain/oe-encoder/_orchestrator/evid"
RES=154; EPOCHS=list(range(19,28)); NHELD=8192; CHUNK=4096
t0=time.time()
def log(s): print(f"[{time.time()-t0:6.0f}s] {s}", flush=True)
RESULTS=f"{OUT}/results.jsonl"
done={json.loads(l)["tag"] for l in open(RESULTS)} if os.path.exists(RESULTS) else set()
def record(d):
    with open(RESULTS,"a") as f: f.write(json.dumps(d)+"\n")
    log("RESULT "+json.dumps(d))

em=load_eval_model(f"{RUN}/checkpoints/epoch_{EPOCHS[0]}", image_size=RES)
model=em.model.cuda().eval(); tok=em.tokenizer
mkeys=set(model.state_dict().keys())

X,Y=torch.load(f"/weka/oe-mm-pretrain/oe-encoder/_orchestrator/regress/imagenet_val_r{RES}.pt")
log(f"imagenet val cached {tuple(X.shape)}")

# ---- fixed held-out training batch (built once, reused for every checkpoint) ----
held=f"{OUT}/held_english_r{RES}_{NHELD}.pt"
if not os.path.exists(held):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    norm=[t for t in em.preprocess_val.transforms if isinstance(t, T.Normalize)][0]
    pre=T.Compose([T.Resize(RES, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(RES),
                   T.Lambda(lambda im: im.convert("RGB")), T.ToTensor(), norm])
    args=parse_args(["--train-data","/weka/oe-mm-pretrain/oe-encoder/english_packed/0190::/weka/oe-mm-pretrain/oe-encoder/english_packed/0191",
        "--dataset-type","webdataset","--text-target","sample_k_positives_from_original_and_dense",
        "--original-caption-prob","0.5","--text-subset-size","5","--no-text-sample-replace",
        "--strip-caption-boilerplate","--train-num-samples",str(NHELD),"--model",em.model_name,
        "--batch-size","256","--workers","8","--epochs","1"])
    args.distributed=False; args.world_size=1; args.rank=0
    di=D.get_wds_dataset(args, pre, is_train=True, tokenizer=tok)
    ims,txs=[],[]; n=0
    for bt in di.dataloader:
        im=bt["image"] if isinstance(bt,dict) else bt[0]; tx=bt["text"] if isinstance(bt,dict) else bt[1]
        ims.append(im.half()); txs.append(tx); n+=im.shape[0]
        log(f"  held-out build {n}/{NHELD}")
        if n>=NHELD: break
    torch.save((torch.cat(ims)[:NHELD], torch.cat(txs)[:NHELD]), held)
HI,HT=torch.load(held); log(f"held-out {tuple(HI.shape)} txt {tuple(HT.shape)}")

def _vec(out):
    if isinstance(out, dict):
        c=[v for v in out.values() if torch.is_tensor(v) and v.dim()==2]
        assert len(c)==1, sorted(out); return c[0]
    return out

@torch.no_grad()
def imagenet():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clf=build_zero_shot_classifier(model, tokenizer=tok, classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES, num_classes_per_batch=10, device="cuda", use_tqdm=False)
    c1=c5=0
    for i in range(0,len(X),500):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f=F.normalize(_vec(model.encode_image(X[i:i+500].cuda().float())).float(),dim=-1)
        top=(f@clf.float()).topk(5,-1).indices.cpu(); y=Y[i:i+500]
        c1+=(top[:,0]==y).sum().item(); c5+=(top==y[:,None]).any(1).sum().item()
    return 100*c1/len(X), 100*c5/len(X)

@torch.no_grad()
def heldout():
    fi,ft=[],[]
    for i in range(0,NHELD,512):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fi.append(F.normalize(_vec(model.encode_image(HI[i:i+512].cuda().float())).float(),dim=-1))
            tx=HT[i:i+512]; K=tx.shape[1]
            t=_vec(model.encode_text(tx.reshape(-1,tx.shape[-1]).cuda()))
            ft.append(F.normalize(t.float(),dim=-1).reshape(-1,K,t.shape[-1]))
    fi=torch.cat(fi); ft=torch.cat(ft); K=ft.shape[1]
    s=model.logit_scale.exp().float().clamp(max=100); b=model.logit_bias.float()
    losses=[]; r1=0
    for i in range(0,NHELD,CHUNK):   # per-GPU-sized chunks, multi-positive SigLIP as in training
        I=fi[i:i+CHUNK]; Tt=ft[i:i+CHUNK].reshape(-1,ft.shape[-1]); B=I.shape[0]
        lg=s*I@Tt.T+b
        lab=-torch.ones_like(lg); idx=torch.arange(B,device=lg.device)
        for k in range(K): lab[idx, idx*K+k]=1
        losses.append((-F.logsigmoid(lab*lg).sum()/B/K).item())
        sim=(I@ft[i:i+CHUNK][:,0].T)   # image->first caption retrieval
        r1+=(sim.argmax(1)==idx).sum().item()
    return float(np.mean(losses)), 100*r1/NHELD

def evaluate(tag, extra):
    if tag in done: log(f"skip {tag} (done)"); return
    t1,t5=imagenet(); l,r=heldout()
    record(dict(tag=tag, imagenet_top1=round(t1,2), imagenet_top5=round(t5,2),
                held_loss=round(l,5), held_i2t_r1=round(r,2),
                logit_scale=round(model.logit_scale.exp().item(),3), logit_bias=round(model.logit_bias.item(),4), **extra))

def filt(sd): return {k:v.float() for k,v in sd.items() if k in mkeys and torch.is_tensor(v) and v.is_floating_point()}

W={}
for e in EPOCHS:
    W[e]=filt(load_checkpoint_state(f"{RUN}/checkpoints/epoch_{e}"))
    log(f"loaded epoch_{e} ({len(W[e])} tensors)")
    model.load_state_dict(W[e], strict=False); evaluate(f"epoch_{e}", {"kind":"single","epochs":[e]})

# ---- T3 geometry (no GPU eval needed) ----
if "geometry" not in done:
    keys=[k for k in W[EPOCHS[0]] if W[EPOCHS[0]][k].dim()>=2]   # weight matrices only (Lamb trust ratio applies)
    def flat(e): return torch.cat([W[e][k].flatten() for k in keys]).cuda()
    V={e:flat(e) for e in EPOCHS}
    dlt={e:V[e+1]-V[e] for e in EPOCHS[:-1]}
    cos={f"{e}->{e+1} vs {e+1}->{e+2}": round(F.cosine_similarity(dlt[e],dlt[e+1],dim=0).item(),4) for e in EPOCHS[:-2]}
    dist={f"gap{g}": round(float(np.mean([(V[e+g]-V[e]).norm().item() for e in EPOCHS if e+g in V])),3) for g in (1,2,3,4,6,8)}
    wnorm=round(V[EPOCHS[0]].norm().item(),2)
    # per block cosine for the dip pair
    blk={}
    for k in keys:
        a=W[22][k]-W[21][k]; b_=W[23][k]-W[22][k]
        name=".".join(k.split(".")[:4]); blk.setdefault(name,[]).append((a.flatten(),b_.flatten()))
    blkcos={n:round(F.cosine_similarity(torch.cat([x for x,_ in v]),torch.cat([y for _,y in v]),dim=0).item(),3) for n,v in blk.items()}
    record(dict(tag="geometry", consecutive_update_cosine=cos, mean_distance_by_gap=dist, weight_norm=wnorm,
                blockwise_cos_21_22_vs_22_23=blkcos))
    del V, dlt

# ---- T1 averaging windows + T2 interpolation ----
def avg(es): return {k: sum(W[e][k] for e in es)/len(es) for k in W[es[0]]}
for es in ([19,20,21],[20,21,22],[21,22,23],[22,23,24],[23,24,25],[24,25,26],[25,26,27]):
    tag="avg_"+"_".join(map(str,es))
    if tag in done: log(f"skip {tag}"); continue
    model.load_state_dict(avg(es), strict=False); evaluate(tag, {"kind":"avg","epochs":es})
for al in (0.25,0.5,0.75):
    tag=f"interp_22_23_a{al}"
    if tag in done: continue
    model.load_state_dict({k:(1-al)*W[22][k]+al*W[23][k] for k in W[22]}, strict=False)
    evaluate(tag, {"kind":"interp","alpha":al})
log("DONE")
