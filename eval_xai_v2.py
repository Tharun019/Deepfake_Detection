"""Grad-CAM fix check. Pre-declared: select variant on 50 held-aside fakes by lowest deletion AUC
(keep current if none better), then evaluate once on the original 100 fake + 50 real test images."""
import os, glob, random, csv, statistics as st
import numpy as np, torch, torch.nn.functional as Fn
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from modules.content import analyzer as A
torch.manual_seed(0); OUT = "results/xai_eval"
model = A._get_image_model(); model.eval()
print("conv_head:", model.conv_head, "\nbn2:", model.bn2, flush=True)
FR = [0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0]; N = 224 * 224
def prep(p):
    img, _ = A._crop_to_face_info(Image.open(p).convert("RGB")); return A._transform(img).unsqueeze(0)
def P(xb):
    with torch.no_grad(): return torch.softmax(model(xb), 1)
def cam_map(x, cls, layer, plus=False):
    s = {}
    h1 = layer.register_forward_hook(lambda m, i, o: s.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: s.__setitem__("g", go[0]))
    with torch.enable_grad():
        out = model(x); model.zero_grad(); out[0, cls].backward()
    h1.remove(); h2.remove(); a, g = s["a"][0], s["g"][0]
    if plus:
        g2, g3 = g ** 2, g ** 3; sa = a.sum((1, 2))[:, None, None]
        al = g2 / (2 * g2 + sa * g3 + 1e-7); al = torch.where(g != 0, al, torch.zeros_like(al))
        w = (Fn.relu(g) * al).sum((1, 2))
    else: w = g.mean((1, 2))
    c = Fn.relu((w[:, None, None] * a).sum(0))
    return Fn.interpolate(c[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach()
def occlusion(x, bl, cls, p0):
    ms = []
    for i in range(7):
        for j in range(7):
            m = torch.zeros(1, 1, 224, 224); m[..., i*32:(i+1)*32, j*32:(j+1)*32] = 1; ms.append(m)
    m = torch.cat(ms); s = (p0 - P(x * (1 - m) + bl * m)[:, cls]).reshape(7, 7)
    return Fn.interpolate(s[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0], s
def curves(x, bl, smap, cls):
    order = torch.argsort(smap.flatten() + 1e-9 * torch.rand(N), descending=True); dl, ins = [], []
    for f in FR:
        m = torch.zeros(N); m[order[:int(f * N)]] = 1; m = m.reshape(1, 1, 224, 224)
        dl.append(x * (1 - m) + bl * m); ins.append(bl * (1 - m) + x * m)
    d = P(torch.cat(dl))[:, cls].numpy(); i = P(torch.cat(ins))[:, cls].numpy()
    return float(np.trapezoid(d, FR)), float(np.trapezoid(i, FR)), float(d[FR.index(.2)])
def rank(v): return np.argsort(np.argsort(v.flatten()))
def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def top_iou(a, b, q=.2):
    k = int(q * N); A_ = set(torch.argsort(a.flatten(), descending=True)[:k].tolist()); B_ = set(torch.argsort(b.flatten(), descending=True)[:k].tolist())
    return len(A_ & B_) / len(A_ | B_)
VAR = {"current_convhead": (model.conv_head, False), "gradcam_bn2": (model.bn2, False), "gradcampp_bn2": (model.bn2, True)}
rv = "datasets/rvf10k/rvf10k"
fakes = sorted(glob.glob(rv + "/**/valid/fake/*", recursive=True)); reals = sorted(glob.glob(rv + "/**/valid/real/*", recursive=True))
R = random.Random(0); R.shuffle(fakes); R.shuffle(reals)
def correct(files, cls, skip, need):   # same order as eval_xai.py; skip the first `skip` correct ones
    out, c = [], 0
    for p in files:
        x = prep(p); p0 = float(P(x)[0, cls])
        if p0 < 0.5: continue
        c += 1
        if c > skip: out.append((p, x, p0))
        if len(out) == need: break
    return out
# ---- step 1: selection on 50 held-aside fakes
sel = correct(fakes, 0, 100, 50); print(f"selection images: {len(sel)}", flush=True)
sel_auc = {k: [] for k in VAR}
for p, x, p0 in sel:
    bl = gaussian_blur(x, [51, 51], [20., 20.])
    for k, (L, plus) in VAR.items(): sel_auc[k].append(curves(x, bl, cam_map(x, 0, L, plus), 0)[0])
print("\n=== STEP 1: selection (50 held-aside fakes, deletion AUC, lower = better) ===")
for k in VAR: print(f"  {k:17s}: {st.mean(sel_auc[k]):.3f}")
best = min(VAR, key=lambda k: st.mean(sel_auc[k]))
if st.mean(sel_auc[best]) >= st.mean(sel_auc["current_convhead"]): best = "current_convhead"
print(f"  CHOSEN: {best}", flush=True)
# ---- step 2: one-time test on the original 100 fake + 50 real
rows = []
for group, files, cls, need in (("fake", fakes, 0, 100), ("real", reals, 1, 50)):
    for p, x, p0 in correct(files, cls, 0, need):
        bl = gaussian_blur(x, [51, 51], [20., 20.]); occ, occ7 = occlusion(x, bl, cls, p0)
        maps = {"current_convhead": cam_map(x, cls, *VAR["current_convhead"]), "chosen": cam_map(x, cls, *VAR[best]), "occlusion": occ, "random": torch.rand(224, 224)}
        rec = {"group": group, "file": os.path.basename(p)}
        for k, mp in maps.items():
            da, ia, p20 = curves(x, bl, mp, cls); rec.update({f"{k}_del": da, f"{k}_ins": ia, f"{k}_p20": p20})
        c7 = Fn.adaptive_avg_pool2d(maps["chosen"][None, None], 7)[0, 0]
        rec["chosen_spearman"] = spearman(c7.numpy(), occ7.numpy()); rec["chosen_iou"] = top_iou(maps["chosen"], occ)
        rows.append(rec)
    print(f"{group} done", flush=True)
with open(f"{OUT}/xai_v2_scores.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n=== STEP 2: one-time test (chosen = {best}) ===")
for g in ("fake", "real"):
    rs = [r for r in rows if r["group"] == g]; print(f"\n[{g}] n={len(rs)}")
    for k in ("current_convhead", "chosen", "occlusion", "random"):
        print(f"  {k:17s}: deletion AUC {st.mean(r[k+'_del'] for r in rs):.3f} | insertion AUC {st.mean(r[k+'_ins'] for r in rs):.3f} | conf after deleting top 20% {st.mean(r[k+'_p20'] for r in rs):.3f}")
    print(f"  chosen beats random on deletion {sum(r['chosen_del'] < r['random_del'] for r in rs)}/{len(rs)}, insertion {sum(r['chosen_ins'] > r['random_ins'] for r in rs)}/{len(rs)}")
    print(f"  chosen vs occlusion agreement: Spearman {st.mean(r['chosen_spearman'] for r in rs):.3f} | top-20% IoU {st.mean(r['chosen_iou'] for r in rs):.3f}")
print(f"\nSaved {OUT}/xai_v2_scores.csv")
