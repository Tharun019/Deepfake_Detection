"""Reviewer 5 baseline: official AASIST (clovaai, ASVspoof 2019 LA weights) zero-shot on the
EXACT same ITW test clips (7 unseen speakers, 1500+1500) and ASV dev sample used for our Wav2Vec2.
Clip lists + our scores come from checkpoints/wav2vec2_itw_ft_v1/compare_cache_*.json, so both
models see identical files. AASIST preprocessing = upstream eval: 16 kHz mono, first 64600 samples,
tile-pad if shorter, score = logit[:,1] (higher = bona fide). No tuning, no threshold search."""
import os, sys, json, subprocess, importlib.util, csv, time
import numpy as np, torch
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
AAS = os.path.join(ROOT, "external", "aasist")
CK = os.path.join(ROOT, "checkpoints", "wav2vec2_itw_ft_v1")
OUT = os.path.join(ROOT, "results", "baseline_aasist"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv

if not os.path.exists(os.path.join(AAS, "models", "weights", "AASIST.pth")):
    sys.exit("STOP: external/aasist missing. Run: git clone --depth 1 https://github.com/clovaai/aasist external/aasist")
spec = importlib.util.spec_from_file_location("aasist_model", os.path.join(AAS, "models", "AASIST.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg = json.load(open(os.path.join(AAS, "config", "AASIST.conf")))["model_config"]
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = mod.Model(cfg)
model.load_state_dict(torch.load(os.path.join(AAS, "models", "weights", "AASIST.pth"), map_location="cpu"))
model.eval()
try:
    model.to(dev); model(torch.zeros(1, 64600, device=dev))
except Exception as e:
    print("MPS/GPU failed (", str(e)[:80], ") -> CPU", flush=True); dev = "cpu"; model.to(dev)
print("device:", dev, "| params:", sum(p.numel() for p in model.parameters()), flush=True)

def load(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)

def pad(x, n=64600):  # identical to clovaai data_utils.pad
    if len(x) >= n: return x[:n]
    return np.tile(x, int(n / len(x)) + 1)[:n]

def eer_auc(y, s):  # y: 1 = fake, s: higher = more fake
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(roc_auc_score(y, s))

def metrics(y, s_fake, pred_fake):
    y, s_fake, pred_fake = map(np.asarray, (y, s_fake, pred_fake))
    rr = float(np.mean(pred_fake[y == 0] == 0)); fr = float(np.mean(pred_fake[y == 1] == 1))
    e, a = eer_auc(y, s_fake)
    return dict(n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()), real_recall=rr, fake_recall=fr,
                bal_acc=(rr + fr) / 2, auc=a, eer=e)

ours = json.load(open(os.path.join(CK, "compare_cache_epoch_3.json")))
base = json.load(open(os.path.join(CK, "compare_cache_baseline.json")))
res, t0 = {}, time.time()
for split in ("test", "asv"):
    items = ours[split]
    if SMOKE: items = [i for i in items if i[1] == 0][:10] + [i for i in items if i[1] == 1][:10]
    rows, fails = [], 0
    for k, (p, lab, _) in enumerate(items):
        try: x = torch.tensor(pad(load(p)))[None].to(dev)
        except Exception as e: fails += 1; print("load fail", p, e); continue
        with torch.no_grad(): _, out = model(x)
        out = out[0].float().cpu().numpy()
        rows.append([p, lab, float(out[1]), int(out[0] > out[1])])  # bona-fide logit, pred_fake (argmax)
        if (k + 1) % 500 == 0: print(f"  {split}: {k+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, f"aasist_{split}_scores.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([["path", "label", "bonafide_logit", "pred_fake"]] + rows)
    keep = {r[0] for r in rows}
    y = [r[1] for r in rows]
    res[split] = {"AASIST (zero-shot, ASVspoof19 LA)": metrics(y, [-r[2] for r in rows], [r[3] for r in rows])}
    for name, c in (("Ours stage 1 (ASVspoof19 only)", base), ("Ours stage 2 (deployed, epoch 3)", ours)):
        cc = [i for i in c[split] if i[0] in keep]
        res[split][name] = metrics([i[1] for i in cc], [i[2] for i in cc], [int(i[2] >= 0.5) for i in cc])
    res[split]["_load_failures"] = fails

names = {"test": "ITW TEST, 7 unseen speakers (same clips for all models)", "asv": "ASVspoof 2019 LA dev sample (sanity: AASIST should be near-perfect here)"}
print()
for split, r in res.items():
    print("=== " + names[split] + f" | load failures {r['_load_failures']} ===")
    print(f"{'model':36s} {'n':>11s} {'real->REAL':>10s} {'fake->FAKE':>10s} {'bal.acc':>8s} {'AUC':>6s} {'EER':>6s}")
    for m, v in r.items():
        if m.startswith("_"): continue
        print(f"{m:36s} {v['n_real']:>5d}+{v['n_fake']:<5d} {100*v['real_recall']:>9.1f}% {100*v['fake_recall']:>9.1f}% "
              f"{100*v['bal_acc']:>7.1f}% {v['auc']:>6.3f} {100*v['eer']:>5.1f}%")
    print()
json.dump(res, open(os.path.join(OUT, "smoke.json" if SMOKE else "aasist_vs_ours.json"), "w"), indent=1)
print("saved to", OUT, "| total", f"{time.time()-t0:.0f}s")
