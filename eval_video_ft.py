import os, sys, csv, json, time, argparse
import numpy as np
sys.path.insert(0, os.getcwd())
import modules.content.video_analyzer as va
from transformers import VideoMAEModel

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["val", "test"], required=True)
args = ap.parse_args()

FT = "models/video/vmae_ft_v1/best"
BB = os.path.join(FT, "videomae-base-ft")
for p in [BB, os.path.join(FT, "video_head_ft.pth"), os.path.join(FT, "feature_norm_stats_facecrop.npz")]:
    assert os.path.exists(p), f"missing: {p}"

# point the LIVE analyzer code at the fine-tuned files (live site files are untouched)
va.LOCAL_BACKBONE_DIR = BB
va.HEAD_CHECKPOINT = os.path.join(FT, "video_head_ft.pth")
va.NORM_STATS_CHECKPOINT = os.path.join(FT, "feature_norm_stats_facecrop.npz")

# ---- load check: fine-tuned weights must load with nothing missing ----
m = VideoMAEModel.from_pretrained(BB)
sd = m.state_dict()
top = sorted({int(k.split(".")[2]) for k in sd if k.startswith("encoder.layer.")})[-1]
qb_top = float(sd[f"encoder.layer.{top}.attention.attention.query.bias"].abs().sum())
qb_0 = float(sd["encoder.layer.0.attention.attention.query.bias"].abs().sum())
print(f"load check: layer {top} query.bias abs-sum {qb_top:.4f} (trained, must be > 0) | layer 0 query.bias abs-sum {qb_0:.4f} (frozen)")
assert qb_top > 0, "fine-tuned biases did not load"
del m, sd

KAGGLE_VAL = {"bal_acc": 0.801, "real": 0.733, "fake": 0.869, "auc": 0.890}
TOL = 0.02   # pre-declared: local val must be within 2 points of Kaggle val
VAL_CSV, TEST_CSV = "results/video_ft_v1_val.csv", "results/video_ft_v1_test.csv"
OUT = VAL_CSV if args.split == "val" else TEST_CSV

def metrics(path):
    rows = [r for r in csv.DictReader(open(path)) if r["raw_fake_prob"] != "" and not r["error"]]
    y = np.array([int(r["label"]) for r in rows]); p = np.array([float(r["raw_fake_prob"]) for r in rows])
    rr = ((p < 0.5) & (y == 1)).sum() / (y == 1).sum(); fr = ((p >= 0.5) & (y == 0)).sum() / (y == 0).sum()
    from sklearn.metrics import roc_auc_score
    return {"n": len(rows), "real": rr, "fake": fr, "bal_acc": (rr + fr) / 2, "auc": roc_auc_score((y == 0).astype(int), p),
            "real_n": f"{((p < 0.5) & (y == 1)).sum()}/{(y == 1).sum()}", "fake_n": f"{((p >= 0.5) & (y == 0)).sum()}/{(y == 0).sum()}"}

if args.split == "test":
    assert os.path.exists(VAL_CSV), "run --split val first"
    v = metrics(VAL_CSV)
    assert v["n"] == 900, f"val incomplete ({v['n']}/900) — finish --split val first"
    assert abs(v["bal_acc"] - KAGGLE_VAL["bal_acc"]) <= TOL, "local val does not reproduce Kaggle val — do NOT run test, report to Claude"

entries = json.load(open("results/video_split_manifest.json"))[args.split]
done = set()
if os.path.exists(OUT):
    done = {r["path"] for r in csv.DictReader(open(OUT))}
todo = [e for e in entries if e["path"] not in done]
print(f"{args.split}: {len(entries)} clips | done {len(done)} | to run {len(todo)}", flush=True)

fh = open(OUT, "a", newline=""); w = csv.writer(fh)
if not done: w.writerow(["path", "method", "label", "raw_fake_prob", "score", "error"])
t0 = time.time()
for i, e in enumerate(todo, 1):
    r = va.analyze_video(os.path.join("datasets/ff_c23/FaceForensics++_C23", e["path"]))
    f = r.get("features", {})
    w.writerow([e["path"], e["method"], e["label"], f.get("raw_fake_prob", ""), r.get("score", ""), r.get("error", "")]); fh.flush()
    if i % 50 == 0 or i == len(todo):
        s = (time.time() - t0) / i
        print(f"{i}/{len(todo)}  {s:.2f} s/clip  ETA {s*(len(todo)-i)/60:.1f} min", flush=True)
fh.close()

m = metrics(OUT)
print(f"\n===== {args.split.upper()} (local, fine-tuned model) =====")
print(f"scored {m['n']}/{len(entries)} | real recall {m['real_n']} = {m['real']*100:.1f}% | fake recall {m['fake_n']} = {m['fake']*100:.1f}%")
print(f"BALANCED ACC {m['bal_acc']*100:.1f}% | AUC {m['auc']:.3f}")
if args.split == "val":
    d = (m["bal_acc"] - KAGGLE_VAL["bal_acc"]) * 100
    print(f"Kaggle val: 80.1% | difference {d:+.1f} pts -> {'REPRODUCED (ok to run test)' if abs(d) <= TOL*100 else 'NOT REPRODUCED - stop, do not run test'}")
else:
    ok = m["bal_acc"] >= 0.700 and m["real"] >= 0.547
    print("STOP RULE: test bal acc >= 70.0% AND real recall >= 54.7%")
    print(f"  bal acc {m['bal_acc']*100:.1f}% {'PASS' if m['bal_acc'] >= 0.700 else 'FAIL'} | real recall {m['real']*100:.1f}% {'PASS' if m['real'] >= 0.547 else 'FAIL'}")
    print(f"  VERDICT: {'ADOPT the fine-tuned model' if ok else 'REJECT - keep current model (67.4%)'}")
