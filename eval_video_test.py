import os, sys, csv, time, json
sys.path.insert(0, os.getcwd())
from modules.content.video_analyzer import analyze_video

ROOT = "datasets/ff_c23/FaceForensics++_C23"
OUT = "results/video_test_eval.csv"
test = json.load(open("results/video_split_manifest.json"))["test"]

done = set()
if os.path.exists(OUT):
    with open(OUT) as fh:
        done = {r["path"] for r in csv.DictReader(fh)}
todo = [e for e in test if e["path"] not in done]
print(f"test clips: {len(test)} | already done: {len(done)} | to run: {len(todo)}", flush=True)

new = not os.path.exists(OUT)
fh = open(OUT, "a", newline="")
w = csv.writer(fh)
if new:
    w.writerow(["path", "method", "label", "raw_fake_prob", "score", "error"])
t0 = time.time()
for i, e in enumerate(todo, 1):
    r = analyze_video(os.path.join(ROOT, e["path"]))
    f = r.get("features", {})
    w.writerow([e["path"], e["method"], e["label"], f.get("raw_fake_prob", ""), r.get("score", ""), r.get("error", "")])
    fh.flush()
    if i % 10 == 0 or i == len(todo):
        s = (time.time() - t0) / i
        print(f"{i}/{len(todo)}  {s:.2f} s/clip  ETA {s*(len(todo)-i)/60:.1f} min", flush=True)
fh.close()
print("DONE")
