# ===================== DeepSentinel — VideoMAE partial fine-tuning (Kaggle, GPU T4 x2) =====================
# PRE-DECLARED STOP RULE (fixed BEFORE training, do not change after seeing results):
#   1. Best epoch is chosen by VALIDATION balanced accuracy only (fake if fake_prob >= 0.5).
#   2. The 900-clip TEST split is NOT used here. It is run ONCE, locally, with eval_video_test.py.
#   3. ADOPT only if  test balanced acc >= 70.0%  AND  test real recall >= 54.7%.
#      Otherwise keep the current frozen model (67.4%) and report this run as a negative result.
# Starting point = the CURRENT deployed model (same head + same mu/sigma), so "epoch 0" on val = today's model.
# ==============================================================================================================

import os, json, time, random, shutil
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import VideoMAEModel, VideoMAEImageProcessor
from sklearn.metrics import roc_auc_score

T_START = time.time()
CONFIG = dict(
    MODEL_ID="MCG-NJU/videomae-base", NUM_FRAMES=16, UNFREEZE_LAST=4, EPOCHS=5, BATCH=16,
    LR_BACKBONE=1e-5, LR_HEAD=1e-4, WD=0.01, SEED=42, MAX_HOURS=10.5, HFLIP=0.5,
)
C = CONFIG
CACHE = "/tmp/vmae_cache"                 # NOT in /kaggle/working (19.5 GiB output limit)
OUT = "/kaggle/working/vmae_ft_v1"
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
random.seed(C["SEED"]); np.random.seed(C["SEED"]); torch.manual_seed(C["SEED"])
DEVICE = torch.device("cuda")
cv2.setNumThreads(0)

def hours(): return (time.time() - T_START) / 3600
free_gb = shutil.disk_usage("/tmp").free / 1e9
print(f"GPUs: {torch.cuda.device_count()} | CPUs: {os.cpu_count()} | free disk /tmp: {free_gb:.1f} GB")
assert free_gb > 30, "Need >30 GB free for the frame cache"

# ---------- locate inputs ----------
def find_dir(pred):
    for root, dirs, files in os.walk("/kaggle/input"):
        if pred(root, dirs, files):
            return root
    return None
DATASET_ROOT = find_dir(lambda r, d, f: "original" in d and "Deepfakes" in d)
ASSET_DIR = find_dir(lambda r, d, f: "video_head_facecrop.pth" in f)
assert DATASET_ROOT, "FF++ dataset not attached"
assert ASSET_DIR, "Upload dataset with video_head_facecrop.pth, feature_norm_stats_facecrop.npz, video_split_manifest.json"
manifest = json.load(open(os.path.join(ASSET_DIR, "video_split_manifest.json")))
counts = {k: len(v) for k, v in manifest.items()}
print("Dataset:", DATASET_ROOT, "| assets:", ASSET_DIR, "| split:", counts)
assert counts == {"train": 4200, "val": 900, "test": 900}, "Manifest does not match the documented split"

# ---------- preprocessing: copied verbatim from training notebook cell 3 (= live video_analyzer.py) ----------
_cascade = None
def crop_to_face_np(frame_rgb):
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    try:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return frame_rgb
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        mx, my = int(w * 0.2), int(h * 0.2)
        ih, iw = frame_rgb.shape[:2]
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(iw, x + w + mx), min(ih, y + h + my)
        return frame_rgb[y0:y1, x0:x1]
    except Exception:
        return frame_rgb

def extract_frames(video_path, num_frames=C["NUM_FRAMES"]):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None
    target = set(np.linspace(0, max(total - 1, 0), num_frames, dtype=int).tolist())
    frames, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok: break
        if idx in target:
            ok2, frame = cap.retrieve()
            if ok2:
                frames.append(crop_to_face_np(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
        if len(frames) >= num_frames: break
    cap.release()
    if len(frames) < num_frames:
        if not frames: return None
        while len(frames) < num_frames: frames.append(frames[-1])
    return frames[:num_frames]

processor = VideoMAEImageProcessor.from_pretrained(C["MODEL_ID"])   # download once in main process

# ---------- stage 1: cache processed clips (train + val only; test stays local) ----------
_proc = None
class CacheBuilder(Dataset):
    def __init__(self, split): self.split, self.entries = split, manifest[split]
    def __len__(self): return len(self.entries)
    def __getitem__(self, i):
        global _proc
        dst = f"{CACHE}/{self.split}_{i}.pt"
        if os.path.exists(dst): return i, 1
        if _proc is None: _proc = VideoMAEImageProcessor.from_pretrained(C["MODEL_ID"])
        frames = extract_frames(os.path.join(DATASET_ROOT, self.entries[i]["path"]))
        if frames is None: return i, 0
        px = _proc(images=frames, return_tensors="pt")["pixel_values"]
        if px.dim() == 5: px = px[0]
        torch.save(px.half().clone(), dst)
        return i, 1

def build_cache(split):
    dl = DataLoader(CacheBuilder(split), batch_size=None, num_workers=os.cpu_count())
    ok, t0, n_all = [], time.time(), len(manifest[split])
    for n, (i, good) in enumerate(dl, 1):
        if int(good): ok.append(int(i))
        if n % 200 == 0 or n == n_all:
            r = n / (time.time() - t0)
            print(f"[cache {split}] {n}/{n_all} — {r:.2f} clips/s — ETA {(n_all-n)/r/60:.1f} min", flush=True)
    print(f"[cache {split}] done — {len(ok)} ok, {n_all-len(ok)} failed")
    return sorted(ok)

train_ok = build_cache("train")
val_ok = build_cache("val")
print(f"Cache finished at {hours():.2f} h")

# ---------- datasets ----------
class ClipDS(Dataset):
    def __init__(self, split, idxs, train):
        self.items = [(f"{CACHE}/{split}_{i}.pt", manifest[split][i]["label"]) for i in idxs]
        self.train = train
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        p, y = self.items[k]
        x = torch.load(p).float()                       # (16, 3, 224, 224)
        if self.train and torch.rand(1).item() < C["HFLIP"]:
            x = torch.flip(x, dims=[3])                  # same horizontal flip on every frame
        return x, y

train_ds, val_ds = ClipDS("train", train_ok, True), ClipDS("val", val_ok, False)
labels = np.array([y for _, y in train_ds.items])
w = np.where(labels == 1, 1.0 / (labels == 1).sum(), 1.0 / (labels == 0).sum())
sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(train_ds),
                                replacement=True, generator=torch.Generator().manual_seed(C["SEED"]))
nw = os.cpu_count()
train_dl = DataLoader(train_ds, batch_size=C["BATCH"], sampler=sampler, num_workers=nw, pin_memory=True,
                      drop_last=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=C["BATCH"], shuffle=False, num_workers=nw, pin_memory=True)

# ---------- model: current deployed model as the starting point ----------
class Head(nn.Module):   # identical to live _Head
    def __init__(self, in_dim=768, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, num_classes))
    def forward(self, x): return self.net(x)

class FT(nn.Module):
    def __init__(self, backbone, head, mu, sigma):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.register_buffer("mu", torch.from_numpy(mu.astype(np.float32)))
        self.register_buffer("sigma", torch.from_numpy(sigma.astype(np.float32)))
    def forward(self, px):
        emb = self.backbone(pixel_values=px).last_hidden_state.mean(dim=1).float()   # same pooling as live
        return self.head((emb - self.mu) / self.sigma)

backbone = VideoMAEModel.from_pretrained(C["MODEL_ID"])
head = Head()
head.load_state_dict(torch.load(os.path.join(ASSET_DIR, "video_head_facecrop.pth"), map_location="cpu"))
stats = np.load(os.path.join(ASSET_DIR, "feature_norm_stats_facecrop.npz"))
model = FT(backbone, head, stats["mu"], stats["sigma"])

L = backbone.config.num_hidden_layers
train_prefixes = [f"encoder.layer.{i}." for i in range(L - C["UNFREEZE_LAST"], L)]
for n, p in model.backbone.named_parameters():
    p.requires_grad = any(n.startswith(t) for t in train_prefixes) or n.startswith("layernorm") or n.startswith("fc_norm")
n_bb = sum(p.numel() for p in model.backbone.parameters())
n_bb_tr = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
n_head = sum(p.numel() for p in model.head.parameters())
print(f"Encoder params: {n_bb/1e6:.2f}M | trainable in encoder: {n_bb_tr/1e6:.2f}M (top {C['UNFREEZE_LAST']} of {L} blocks) | head: {n_head:,}")

model.to(DEVICE)
net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
opt = torch.optim.AdamW([
    {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": C["LR_BACKBONE"]},
    {"params": list(model.head.parameters()), "lr": C["LR_HEAD"]},
], weight_decay=C["WD"])
steps_per_epoch = len(train_dl)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C["EPOCHS"] * steps_per_epoch)
scaler = torch.amp.GradScaler("cuda")
crit = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate():
    net.eval()
    P, Y = [], []
    for x, y in val_dl:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(x.to(DEVICE, non_blocking=True))
        P += torch.softmax(logits.float(), dim=1)[:, 0].cpu().tolist(); Y += y.tolist()
    net.train()
    P, Y = np.array(P), np.array(Y)
    rr = float(((P < 0.5) & (Y == 1)).sum() / (Y == 1).sum())
    fr = float(((P >= 0.5) & (Y == 0)).sum() / (Y == 0).sum())
    return {"real_recall": rr, "fake_recall": fr, "bal_acc": (rr + fr) / 2,
            "auc": float(roc_auc_score((Y == 0).astype(int), P)), "n": int(len(Y))}

def save_best(ep, m):
    d = os.path.join(OUT, "best")
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    model.backbone.save_pretrained(os.path.join(d, "videomae-base-ft"))
    processor.save_pretrained(os.path.join(d, "videomae-base-ft"))
    torch.save(model.head.state_dict(), os.path.join(d, "video_head_ft.pth"))
    np.savez(os.path.join(d, "feature_norm_stats_facecrop.npz"), mu=stats["mu"], sigma=stats["sigma"])
    json.dump({"epoch": ep, **m, "config": C}, open(os.path.join(d, "metrics.json"), "w"), indent=2)

def fmt(m): return f"bal_acc {m['bal_acc']*100:.1f}% | real {m['real_recall']*100:.1f}% | fake {m['fake_recall']*100:.1f}% | AUC {m['auc']:.3f}"

log = []
base = evaluate()
log.append({"epoch": 0, **base})
print(f"EPOCH 0 (current deployed model) VAL: {fmt(base)}", flush=True)
best = {"epoch": 0, **base}

# ---------- stage 2: fine-tune ----------
net.train(); stopped = False
for ep in range(1, C["EPOCHS"] + 1):
    t_ep, run_loss = time.time(), 0.0
    for step, (x, y) in enumerate(train_dl, 1):
        if hours() > C["MAX_HOURS"]:
            stopped = True; break
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(net(x).float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run_loss += loss.item()
        if step % 50 == 0 or step == steps_per_epoch:
            s = (time.time() - t_ep) / step
            print(f"ep {ep} step {step}/{steps_per_epoch} — loss {run_loss/step:.4f} — {s:.2f} s/step — "
                  f"epoch ETA {(steps_per_epoch-step)*s/60:.1f} min — total {hours():.2f} h", flush=True)
    if stopped:
        print(f"TIME LIMIT reached at {hours():.2f} h — stopping before epoch {ep} finished"); break
    m = evaluate()
    log.append({"epoch": ep, **m, "train_loss": run_loss / steps_per_epoch, "hours": hours()})
    json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"), indent=2)
    tag = ""
    if m["bal_acc"] > best["bal_acc"]:
        best = {"epoch": ep, **m}; save_best(ep, m); tag = "  <-- new best on VAL, saved"
    print(f"EPOCH {ep} VAL: {fmt(m)}{tag}", flush=True)

json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"), indent=2)
print("\n================ SUMMARY (VAL only — test is run locally) ================")
print(f"Epoch 0 (current model): {fmt(base)}")
print(f"Best epoch {best['epoch']}:          {fmt(best)}")
if best["epoch"] == 0:
    print("No fine-tuned epoch beat the current model on VAL -> REJECT, keep the frozen model (67.4%).")
else:
    print(f"Saved: {OUT}/best  -> download it, then run the 900-clip TEST locally ONCE and apply the stop rule.")
print(f"Total time: {hours():.2f} h")
