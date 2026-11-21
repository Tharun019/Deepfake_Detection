"""Validate Grad-CAM vs occlusion (established) and random baseline: deletion/insertion AUC + agreement.
Pre-declared: rvf10k valid, seed 0, 100 correctly-classified fakes + 50 reals, fixed settings."""
import os, glob, random, csv, inspect, statistics as st
import numpy as np, torch, torch.nn.functional as Fn
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from modules.content import analyzer as A
torch.manual_seed(0); OUT = "results/xai_eval"
src = inspect.getsource(A.generate_gradcam); print("system Grad-CAM uses conv_head:", "conv_head" in src, flush=True)
model = A._get_image_model(); model.eval()
layer = model.conv_head; print("target layer:", layer, flush=True)
FR = [0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0]; N = 224 * 224
def prep(p):
    img, face = A._crop_to_face_info(Image.open(p).convert("RGB")); return A._transform(img).unsqueeze(0), face
def P(xb):
    with torch.no_grad(): return torch.softmax(model(xb), 1)
def gradcam(x, cls):
    st_ = {}
    h1 = layer.register_forward_hook(lambda m, i, o: st_.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: st_.__setitem__("g", go[0]))
    with torch.enable_grad():
        out = model(x); model.zero_grad(); out[0, cls].backward()
    h1.remove(); h2.remove()
    a, g = st_["a"][0], st_["g"][0]; cam = Fn.relu((g.mean((1, 2))[:, None, None] * a).sum(0))
    return Fn.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach()
def occlusion(x, bl, cls, p0):
    ms = []
    for i in range(7):
        for j in range(7):
            m = torch.zeros(1, 1, 224, 224); m[..., i*32:(i+1)*32, j*32:(j+1)*32] = 1; ms.append(m)
    m = torch.cat(ms); xb = x * (1 - m) + bl * m
    s = (p0 - P(xb)[:, cls]).reshape(7, 7)
    return Fn.interpolate(s[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0], s
def curves(x, bl, smap, cls):
    order = torch.argsort(smap.flatten() + 1e-9 * torch.rand(N), descending=True); dl, ins = [], []
    for f in FR:
        m = torch.zeros(N); m[order[:int(f * N)]] = 1; m = m.reshape(1, 1, 224, 224)
        dl.append(x * (1 - m) + bl * m); ins.append(bl * (1 - m) + x * m)
    d = P(torch.cat(dl))[:, cls].numpy(); i = P(torch.cat(ins))[:, cls].numpy()
    return d, i, float(np.trapezoid(d, FR)), float(np.trapezoid(i, FR))
def rank(v): return np.argsort(np.argsort(v.flatten()))
def spearman(a, b): ra, rb = rank(a), rank(b); return float(np.corrcoef(ra, rb)[0, 1])
def top_iou(a, b, q=.2):
    k = int(q * N); A_ = set(torch.argsort(a.flatten(), descending=True)[:k].tolist()); B_ = set(torch.argsort(b.flatten(), descending=True)[:k].tolist())
    return len(A_ & B_) / len(A_ | B_)
rv = "datasets/rvf10k/rvf10k"
fakes = sorted(glob.glob(rv + "/**/valid/fake/*", recursive=True)); reals = sorted(glob.glob(rv + "/**/valid/real/*", recursive=True))
R = random.Random(0); R.shuffle(fakes); R.shuffle(reals)
rows, curve_acc = [], {}
for group, files, cls, need in (("fake", fakes, 0, 100), ("real", reals, 1, 50)):
    got = 0
    for p in files:
        if got == need: break
        x, face = prep(p); p0 = float(P(x)[0, cls])
        if p0 < 0.5: continue                       # only correctly classified images
        bl = gaussian_blur(x, [51, 51], [20., 20.])
        cam = gradcam(x, cls); occ, occ7 = occlusion(x, bl, cls, p0); rnd = torch.rand(224, 224)
        cam7 = Fn.adaptive_avg_pool2d(cam[None, None], 7)[0, 0]
        rec = [group, os.path.basename(p), face, round(p0, 4), round(spearman(cam7.numpy(), occ7.numpy()), 4), round(top_iou(cam, occ), 4)]
        for name, smap in (("gradcam", cam), ("occlusion", occ), ("random", rnd)):
            d, i, da, ia = curves(x, bl, smap, cls); rec += [round(da, 4), round(ia, 4), round(float(d[FR.index(.2)]), 4)]
            acc = curve_acc.setdefault((group, name), [np.zeros(len(FR)), np.zeros(len(FR)), 0]); acc[0] += d; acc[1] += i; acc[2] += 1
        rows.append(rec); got += 1
        if got % 10 == 0: print(f"{group}: {got}/{need}", flush=True)
hdr = ["group", "file", "face", "p_orig", "spearman_cam_vs_occ", "top20_iou_cam_vs_occ"] + [f"{m}_{k}" for m in ("gradcam", "occlusion", "random") for k in ("del_auc", "ins_auc", "p_after_del20")]
with open(f"{OUT}/xai_scores.csv", "w", newline="") as fh: csv.writer(fh).writerows([hdr] + rows)
with open(f"{OUT}/xai_mean_curves.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["group", "method", "curve"] + [f"f{f}" for f in FR])
    for (g, m), (d, i, n) in curve_acc.items(): w.writerow([g, m, "deletion"] + list(np.round(d / n, 4))); w.writerow([g, m, "insertion"] + list(np.round(i / n, 4)))
print("\n=== XAI RESULTS (deletion AUC: lower = more faithful; insertion AUC: higher = more faithful) ===")
for g in ("fake", "real"):
    rs = [r for r in rows if r[0] == g]
    if not rs: continue
    print(f"\n[{g}] n={len(rs)}  mean original confidence {st.mean(r[3] for r in rs):.3f}")
    for k, name in enumerate(("gradcam", "occlusion", "random")):
        b = 6 + 3 * k
        print(f"  {name:9s}: deletion AUC {st.mean(r[b] for r in rs):.3f} (sd {st.pstdev(r[b] for r in rs):.3f}) | insertion AUC {st.mean(r[b+1] for r in rs):.3f} | confidence after deleting top 20% {st.mean(r[b+2] for r in rs):.3f}")
    print(f"  Grad-CAM beats random on deletion in {sum(r[6] < r[12] for r in rs)}/{len(rs)} images; on insertion in {sum(r[7] > r[13] for r in rs)}/{len(rs)}")
    print(f"  agreement Grad-CAM vs occlusion: mean Spearman {st.mean(r[4] for r in rs):.3f} | mean top-20% IoU {st.mean(r[5] for r in rs):.3f} (random overlap ~0.111)")
print(f"\nSaved {OUT}/xai_scores.csv and {OUT}/xai_mean_curves.csv")
