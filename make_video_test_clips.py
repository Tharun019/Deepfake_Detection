import os, json, random, shutil
from collections import defaultdict

def find_dataset_root():
    for root, dirs, files in os.walk("datasets/ff_c23"):
        if "original" in dirs and "Deepfakes" in dirs:
            return root
    raise FileNotFoundError("dataset not found")
DATASET_ROOT = find_dataset_root()
print("Dataset root:", DATASET_ROOT)

# ---- exact copy of notebook cell 3 split logic ----
FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
REAL_DIR = "original"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
def real_id(f): return os.path.splitext(f)[0]
def parse_pair(f):
    s, t = os.path.splitext(f)[0].split("_"); return s, t

real_ids = sorted(real_id(f) for f in os.listdir(os.path.join(DATASET_ROOT, REAL_DIR)) if f.endswith(".mp4"))
for rid in real_ids: find(rid)
fake_files = defaultdict(list)
for method in FAKE_METHODS:
    for f in os.listdir(os.path.join(DATASET_ROOT, method)):
        if f.endswith(".mp4"):
            fake_files[method].append(f)
            s, t = parse_pair(f); union(s, t)
components = defaultdict(list)
for rid in real_ids: components[find(rid)].append(rid)
comp_list = list(components.items())
random.Random(SEED).shuffle(comp_list)
n_real = len(real_ids)
targets = {k: v * n_real for k, v in SPLIT_RATIOS.items()}
assigned = {"train": [], "val": [], "test": []}
counts = {"train": 0, "val": 0, "test": 0}
for comp_root, ids in comp_list:
    deficits = {k: targets[k] - counts[k] for k in targets}
    best = max(deficits, key=lambda k: deficits[k])
    assigned[best].extend(ids); counts[best] += len(ids)
id_to_split = {rid: s for s, ids in assigned.items() for rid in ids}
manifest = {"train": [], "val": [], "test": []}
for rid in real_ids:
    manifest[id_to_split[rid]].append({"path": os.path.join(REAL_DIR, f"{rid}.mp4"), "label": 1, "method": "original"})
for method in FAKE_METHODS:
    for f in fake_files[method]:
        s, t = parse_pair(f)
        s_split, t_split = id_to_split.get(s), id_to_split.get(t)
        if s_split == t_split:
            manifest[s_split].append({"path": os.path.join(method, f), "label": 0, "method": method})
# ---- end of notebook logic ----

vid_counts = {k: len(v) for k, v in manifest.items()}
id_counts = {k: len(v) for k, v in assigned.items()}
test_real = sum(e["label"] == 1 for e in manifest["test"])
print("Real-id split counts:", id_counts, "| notebook: {'train': 700, 'val': 150, 'test': 150}")
print("Video counts per split:", vid_counts, "| notebook: {'train': 4200, 'val': 900, 'test': 900}")
print(f"test: {test_real} real, {vid_counts['test'] - test_real} fake | notebook: 150 real, 750 fake")
if id_counts != {"train": 700, "val": 150, "test": 150} or vid_counts != {"train": 4200, "val": 900, "test": 900} or test_real != 150:
    raise SystemExit("ABORT: split does NOT match the notebook. Nothing copied.")
print("SPLIT MATCHES NOTEBOOK")

os.makedirs("results", exist_ok=True)
with open("results/video_split_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print("Saved results/video_split_manifest.json")

rng = random.Random(2026)
test = manifest["test"]
picks_real = rng.sample([e for e in test if e["label"] == 1], 10)
picks_fake = []
for m in FAKE_METHODS:
    picks_fake += rng.sample([e for e in test if e["method"] == m], 2)

out = os.path.expanduser("~/Desktop/VIDEOS")
for sub, picks in [("real_video", picks_real), ("fake_video", picks_fake)]:
    d = os.path.join(out, sub); os.makedirs(d, exist_ok=True)
    for e in picks:
        src = os.path.join(DATASET_ROOT, e["path"])
        dst = os.path.join(d, f"{e['method']}__{os.path.basename(e['path'])}")
        shutil.copy2(src, dst)
        print(f"  {sub}/{os.path.basename(dst)}  ({os.path.getsize(dst)/1e6:.1f} MB)")
print("Done:", len(picks_real), "real +", len(picks_fake), "fake copied to", out)
