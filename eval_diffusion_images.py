"""Measure the image model (rvf10k-trained) on diffusion faces. No tuning: report as-is.
Data: OpenRL/DeepFakeFace (HF). wiki = real (IMDB-WIKI), text2img = SD v1.5, inpainting = SD inpainting."""
import io, os, random, zipfile, csv, inspect, sys
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files
from modules.content import analyzer as A

REPO = "OpenRL/DeepFakeFace"; N = 500; SEED = 0
SETS = {"real_wiki": ("wiki.zip", 0), "sd15_text2img": ("text2img.zip", 1), "sd_inpainting": ("inpainting.zip", 1)}
OUT = "results/diffusion_eval"; CACHE = os.path.expanduser("~/Desktop/Deepfake_Detection/datasets/deepfakeface")
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

print("=== check: system image scoring code ===")
print(inspect.getsource(A._analyze_image))
files = list_repo_files(REPO, repo_type="dataset"); print("repo files:", files)
for z, _ in SETS.values():
    if z not in files: sys.exit(f"STOP: {z} not found in {REPO}")

model = A._get_image_model(); model.eval()
def fake_prob(img):
    img, face = A._crop_to_face_info(img.convert("RGB"))
    with torch.no_grad():
        p = torch.softmax(model(A._transform(img).unsqueeze(0)), dim=1).squeeze()
    return float(p[0]), face          # class 0 = fake (project convention)

rows, manifest = [], []
for name, (zname, label) in SETS.items():
    zpath = hf_hub_download(REPO, zname, repo_type="dataset", local_dir=CACHE)
    zf = zipfile.ZipFile(zpath)
    imgs = sorted(m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png")))
    random.Random(SEED).shuffle(imgs)
    picked, skipped = [], 0
    for m in imgs:
        if len(picked) == N: break
        try:
            im = Image.open(io.BytesIO(zf.read(m))); im.load()
            if min(im.size) < 64: skipped += 1; continue
        except Exception: skipped += 1; continue
        fp, face = fake_prob(im)
        picked.append(m); rows.append([name, label, m, round(fp, 4), face])
    manifest += [[name, m] for m in picked]
    print(f"{name}: {len(imgs)} images in zip, scored {len(picked)}, skipped {skipped}")

# sanity: our score vs the live system's _analyze_image on 3 files
print("=== sanity vs _analyze_image (first 3 real) ===")
zf = zipfile.ZipFile(os.path.join(CACHE, "wiki.zip"))
for r in [r for r in rows if r[0] == "real_wiki"][:3]:
    tmp = os.path.join(OUT, "_tmp_" + os.path.basename(r[2])); open(tmp, "wb").write(zf.read(r[2]))
    print(r[3], "->", A._analyze_image(tmp)); os.remove(tmp)

with open(f"{OUT}/scores.csv", "w", newline="") as f:
    csv.writer(f).writerows([["set", "label", "file", "fake_prob", "face_status"]] + rows)
with open(f"{OUT}/manifest.csv", "w", newline="") as f:
    csv.writer(f).writerows([["set", "file"]] + manifest)

def auc(pos, neg):  # probability a fake scores higher than a real
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return s / (len(pos) * len(neg))
real = [r[3] for r in rows if r[0] == "real_wiki"]
real_ok = sum(p < 0.5 for p in real) / len(real)
print("\n=== RESULTS (model-level call, fake if prob >= 0.50) ===")
print(f"real_wiki      : real called real {real_ok*100:.1f}%  (FPR {100-real_ok*100:.1f}%)  n={len(real)}")
for name in ["sd15_text2img", "sd_inpainting"]:
    fk = [r[3] for r in rows if r[0] == name]; det = sum(p >= 0.5 for p in fk) / len(fk)
    print(f"{name:15s}: fakes detected {det*100:.1f}%  balanced acc vs real {(det+real_ok)/2*100:.1f}%  AUC {auc(fk, real):.3f}  n={len(fk)}")
nf = {}
for r in rows: nf.setdefault(r[0], [0, 0]); nf[r[0]][0] += 1; nf[r[0]][1] += (str(r[4]).lower().startswith(("face", "true", "found")))
print("face-status sample values:", sorted({str(r[4]) for r in rows})[:5])
print(f"\nSaved: {OUT}/scores.csv and {OUT}/manifest.csv")
