"""Robustness of L1/L2/L3 + fusion to compression, re-encoding and metadata manipulation.
Pre-declared: fixed variants, fixed samples (seed 0), no tuning; report as-is."""
import os, sys, csv, glob, random, shutil, subprocess, json, io, statistics as st
from PIL import Image, TiffImagePlugin
from modules.metadata.analyzer import analyze as L1
from modules.content.analyzer import analyze as L2
from modules.binary.analyzer import analyze as L3
from modules.scoring.fusion import fuse_scores

OUT = "results/robustness"; TMP = OUT + "/tmp"; os.makedirs(TMP, exist_ok=True)
R = random.Random(0)
def sc(x):  # layer analyzers may return (score, details), dict or float
    if isinstance(x, tuple): x = x[0]
    if isinstance(x, dict): x = x.get("score", x.get("confidence", 0.5))
    return float(x)
def run(path, mt):
    a, b, c = sc(L1(path, mt)), sc(L2(path, mt)), sc(L3(path, mt))
    f = fuse_scores(a, b, c); return a, b, c, f["confidence"], int(f["is_fake"])

# ---------- JPEG byte surgery (no re-compression) ----------
def jpeg_split(b):
    assert b[:2] == b"\xff\xd8"; i, segs = 2, []
    while i < len(b) and b[i] == 0xFF:
        m = b[i+1]
        if m == 0xDA: break
        L = int.from_bytes(b[i+2:i+4], "big"); segs.append((m, b[i:i+2+L])); i += 2 + L
    return segs, b[i:]
def jpeg_strip(b):
    segs, rest = jpeg_split(b)
    return b"\xff\xd8" + b"".join(s for m, s in segs if not (0xE1 <= m <= 0xEF or m == 0xFE)) + rest
def fake_exif_bytes():
    ex = Image.Exif(); ex[271] = "Canon"; ex[272] = "Canon EOS 80D"; ex[305] = "Canon EOS 80D Firmware"; ex[306] = "2024:05:01 10:00:00"
    sub = ex.get_ifd(0x8769)
    sub[33434] = TiffImagePlugin.IFDRational(1, 125); sub[33437] = TiffImagePlugin.IFDRational(56, 10)
    sub[34855] = 200; sub[36867] = "2024:05:01 10:00:00"; sub[37500] = b"\x00" * 32
    p = ex.tobytes(); p = p if p.startswith(b"Exif") else b"Exif\x00\x00" + p
    return b"\xff\xe1" + (len(p) + 2).to_bytes(2, "big") + p
def jpeg_fake_exif(b):
    segs, rest = jpeg_split(jpeg_strip(b))
    app0 = [s for m, s in segs if m == 0xE0]; others = [s for m, s in segs if m != 0xE0]
    return b"\xff\xd8" + b"".join(app0) + fake_exif_bytes() + b"".join(others) + rest

def img_variants(src, label):
    raw = open(src, "rb").read(); isj = raw[:2] == b"\xff\xd8"
    im = Image.open(src); exif = im.info.get("exif"); im = im.convert("RGB"); base = os.path.splitext(os.path.basename(src))[0]
    v = {"original": src}
    for q in (90, 70, 50):
        p = f"{TMP}/{base}_q{q}.jpg"; kw = {"exif": exif} if exif else {}; im.save(p, quality=q, **kw); v[f"jpeg_q{q}"] = p
    p = f"{TMP}/{base}_half.jpg"; kw = {"exif": exif} if exif else {}
    im.resize((max(1, im.width // 2), max(1, im.height // 2))).save(p, quality=90, **kw); v["resize_half_q90"] = p
    if isj:
        p = f"{TMP}/{base}_strip.jpg"; open(p, "wb").write(jpeg_strip(raw)); v["meta_stripped"] = p
        if label == 1:
            p = f"{TMP}/{base}_fakeexif.jpg"; open(p, "wb").write(jpeg_fake_exif(raw)); v["fake_camera_exif"] = p
    return v

# ---------- image files ----------
def find_imgs(root):
    return [f for f in glob.glob(root + "/**/*", recursive=True) if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))]
rv = "datasets/rvf10k/rvf10k"
reals = [f for f in find_imgs(rv) if "/valid/" in f and "/real/" in f]; fakes = [f for f in find_imgs(rv) if "/valid/" in f and "/fake/" in f]
print(f"rvf10k valid found: real {len(reals)}, fake {len(fakes)}", flush=True)
if not reals or not fakes: print("STOP: rvf10k valid/real|fake folders not found; listing:", os.listdir(rv)); sys.exit(1)
R.shuffle(reals); R.shuffle(fakes)
img_items = [(f, 0, "rvf10k") for f in sorted(reals)[:0] or reals[:100]] + [(f, 1, "rvf10k") for f in fakes[:100]]
si = os.path.expanduser("~/Desktop/sample inputs/IMG")
for f in find_imgs(si):
    lab = 0 if "/real" in f.lower() else 1 if "/fake" in f.lower() else None
    if lab is not None: img_items.append((f, lab, "live_test"))
print(f"image items: {len(img_items)}", flush=True)

rows = []
first = True
for k, (f, lab, src) in enumerate(img_items):
    try: vs = img_variants(f, lab)
    except Exception as e: print("skip", f, e); continue
    for vn, p in vs.items():
        try: a, b, c, fu, fk = run(p, "image")
        except Exception as e: print("ERR", vn, f, e); continue
        if first: print("sanity first image:", L1(p, "image"), "|", L2(p, "image"), "|", L3(p, "image"), flush=True); first = False
        rows.append(["image", src, os.path.basename(f), lab, vn, round(a, 4), round(b, 4), round(c, 4), fu, fk])
    if k % 25 == 0: print(f"images {k}/{len(img_items)}", flush=True)
    for p in vs.values():
        if p.startswith(TMP): os.remove(p)

# ---------- video files ----------
idx = {}
for root, _, fs in os.walk("datasets/ff_c23"):
    for x in fs:
        if x.endswith(".mp4"): idx.setdefault(x, []).append(os.path.join(root, x))
tcsv = list(csv.DictReader(open("results/video_ft_v1_test.csv")))
print("video test csv columns:", list(tcsv[0].keys()), flush=True)
def vpath(row):
    for v in row.values():
        if isinstance(v, str) and v.endswith(".mp4"):
            if os.path.exists(v): return v
            c = idx.get(os.path.basename(v), [])
            if len(c) == 1: return c[0]
            m = [x for x in c if all(part in x for part in v.split("/")[-2:-1])]
            if m: return m[0]
    return None
vids = [p for p in (vpath(r) for r in tcsv) if p]
print(f"video test clips resolved: {len(vids)} of {len(tcsv)}", flush=True)
real_v = [p for p in vids if "original" in p.lower()]
methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
R.shuffle(real_v); vitems = [(p, 0) for p in real_v[:50]]
for m in methods:
    mv = [p for p in vids if f"/{m}/" in p]; R.shuffle(mv); vitems += [(p, 1) for p in mv[:10]]
print(f"video items: {len(vitems)}", flush=True)
FF = ["ffmpeg", "-y", "-v", "error"]
def vvar(src):
    b = os.path.splitext(os.path.basename(src))[0]; v = {"original": src}
    for crf in (23, 35):
        p = f"{TMP}/{b}_crf{crf}.mp4"
        subprocess.run(FF + ["-i", src, "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p", "-map_metadata", "0", "-c:a", "aac", p], check=True); v[f"h264_crf{crf}"] = p
    p = f"{TMP}/{b}_strip.mp4"
    subprocess.run(FF + ["-i", src, "-c", "copy", "-map_metadata", "-1", "-fflags", "+bitexact", p], check=True); v["meta_stripped"] = p
    p = f"{TMP}/{b}_fakemeta.mp4"
    subprocess.run(FF + ["-i", src, "-c", "copy", "-map_metadata", "-1", "-fflags", "+bitexact", "-movflags", "use_metadata_tags",
                         "-metadata", "creation_time=2024-05-01T10:00:00Z", "-metadata", "com.apple.quicktime.make=Apple",
                         "-metadata", "com.apple.quicktime.model=iPhone 13", p], check=True); v["fake_phone_meta"] = p
    return v
first = True
for k, (f, lab) in enumerate(vitems):
    try: vs = vvar(f)
    except Exception as e: print("skip", f, e); continue
    for vn, p in vs.items():
        try: a, b, c, fu, fk = run(p, "video")
        except Exception as e: print("ERR", vn, f, e); continue
        if first: print("sanity first video:", L1(p, "video"), "|", L3(p, "video"), flush=True); first = False
        rows.append(["video", "ff_c23", os.path.relpath(f, "datasets/ff_c23"), lab, vn, round(a, 4), round(b, 4), round(c, 4), fu, fk])
    print(f"videos {k+1}/{len(vitems)}", flush=True)
    for p in vs.values():
        if p.startswith(TMP): os.remove(p)

hdr = ["media", "source", "file", "label", "variant", "L1", "L2", "L3", "fused", "pred_fake"]
with open(f"{OUT}/robustness_scores.csv", "w", newline="") as fh: csv.writer(fh).writerows([hdr] + rows)

# ---------- summary ----------
orig = {(r[0], r[1], r[2]): r for r in rows if r[4] == "original"}
print("\n=== RESULTS (fused verdict, threshold 0.35) ===")
for media in ("image", "video"):
    for src in sorted({r[1] for r in rows if r[0] == media}):
        print(f"\n[{media} / {src}]  variant | n | real correct | fakes detected | mean L1 | mean L2 | mean L3 | flips vs original | mean |dL1| | mean |dL3|")
        for vn in dict.fromkeys(r[4] for r in rows if r[0] == media and r[1] == src):
            rs = [r for r in rows if r[0] == media and r[1] == src and r[4] == vn]
            re_ = [r for r in rs if r[3] == 0]; fa = [r for r in rs if r[3] == 1]
            rc = f"{sum(1 - r[9] for r in re_)}/{len(re_)}" if re_ else "-"; fd = f"{sum(r[9] for r in fa)}/{len(fa)}" if fa else "-"
            fl = sum(r[9] != orig[(r[0], r[1], r[2])][9] for r in rs if (r[0], r[1], r[2]) in orig)
            d1 = st.mean(abs(r[5] - orig[(r[0], r[1], r[2])][5]) for r in rs if (r[0], r[1], r[2]) in orig)
            d3 = st.mean(abs(r[7] - orig[(r[0], r[1], r[2])][7]) for r in rs if (r[0], r[1], r[2]) in orig)
            print(f"  {vn:18s} | {len(rs):3d} | {rc:>7s} | {fd:>7s} | {st.mean(r[5] for r in rs):.3f} | {st.mean(r[6] for r in rs):.3f} | {st.mean(r[7] for r in rs):.3f} | {fl:3d} | {d1:.3f} | {d3:.3f}")
print(f"\nSaved {OUT}/robustness_scores.csv")
shutil.rmtree(TMP, ignore_errors=True)
