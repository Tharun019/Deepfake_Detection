"""Audio robustness on the ITW test speakers (unseen). Fixed variants, seed 0, no tuning."""
import os, sys, csv, glob, json, random, subprocess, shutil, statistics as st
from modules.metadata.analyzer import analyze as L1
from modules.content.analyzer import analyze as L2
from modules.binary.analyzer import analyze as L3
from modules.scoring.fusion import fuse_scores
OUT = "results/robustness"; TMP = OUT + "/tmp_audio"; os.makedirs(TMP, exist_ok=True)
def sc(x):
    if isinstance(x, tuple): x = x[0]
    if isinstance(x, dict): x = x.get("score", 0.5)
    return float(x)
def run(p):
    a, b, c = sc(L1(p, "audio")), sc(L2(p, "audio")), sc(L3(p, "audio")); f = fuse_scores(a, b, c)
    return a, b, c, f["confidence"], int(f["is_fake"])
sj = [p for p in glob.glob("**/split.json", recursive=True) if "itw" in p.lower()]
print("split.json found:", sj, flush=True)
if not sj: sys.exit("STOP: ITW split.json not found")
test_spk = set(json.load(open(sj[0]))["test"]); print("test speakers:", sorted(test_spk), flush=True)
root = "datasets/itw/release_in_the_wild"
meta = list(csv.DictReader(open(os.path.join(root, "meta.csv"))))
items = [(os.path.join(root, r["file"]), 0 if r["label"] == "bona-fide" else 1) for r in meta if r["speaker"] in test_spk]
R = random.Random(0); real = [i for i in items if i[1] == 0]; fake = [i for i in items if i[1] == 1]
R.shuffle(real); R.shuffle(fake); sel = real[:50] + fake[:50]
print(f"test clips: real {len(real)}, fake {len(fake)} -> using {len(sel)}", flush=True)
FF = ["ffmpeg", "-y", "-v", "error"]
VARS = [("mp3_128k", ["-c:a", "libmp3lame", "-b:a", "128k"], "mp3"),
        ("mp3_64k", ["-c:a", "libmp3lame", "-b:a", "64k"], "mp3"),
        ("aac_64k", ["-c:a", "aac", "-b:a", "64k"], "m4a"),
        ("phone_8khz", ["-ar", "8000", "-ac", "1"], "wav"),
        ("meta_stripped", ["-map_metadata", "-1", "-fflags", "+bitexact"], "wav"),
        ("fake_recorder_meta", ["-map_metadata", "-1", "-fflags", "+bitexact", "-metadata", "title=Voice Memo 042",
                                "-metadata", "artist=iPhone", "-metadata", "date=2024-05-01",
                                "-metadata", "creation_time=2024-05-01T10:00:00Z"], "wav")]
def variants(src):
    b = os.path.splitext(os.path.basename(src))[0]; v = {"original": src}
    for name, args, ext in VARS:
        p = f"{TMP}/{b}_{name}.{ext}"; subprocess.run(FF + ["-i", src] + args + [p], check=True); v[name] = p
    return v
rows, first = [], True
for k, (f, lab) in enumerate(sel):
    try: vs = variants(f)
    except Exception as e: print("skip", f, e); continue
    for vn, p in vs.items():
        try: a, b, c, fu, fk = run(p)
        except Exception as e: print("ERR", vn, f, e); continue
        if first: print("sanity first audio:", L1(p, "audio"), "|", L2(p, "audio"), "|", L3(p, "audio"), flush=True); first = False
        rows.append(["audio", "itw_test", os.path.basename(f), lab, vn, round(a, 4), round(b, 4), round(c, 4), fu, fk])
    if (k + 1) % 10 == 0: print(f"audio {k+1}/{len(sel)}", flush=True)
    for p in vs.values():
        if p.startswith(TMP): os.remove(p)
with open(f"{OUT}/robustness_audio_scores.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([["media", "source", "file", "label", "variant", "L1", "L2", "L3", "fused", "pred_fake"]] + rows)
orig = {r[2]: r for r in rows if r[4] == "original"}
print("\n=== AUDIO RESULTS (fused verdict, threshold 0.35) ===")
print("variant | n | real correct | fakes detected | mean L1 | mean L2 | mean L3 | flips vs original | mean |dL1| | mean |dL3|")
for vn in dict.fromkeys(r[4] for r in rows):
    rs = [r for r in rows if r[4] == vn]; re_ = [r for r in rs if r[3] == 0]; fa = [r for r in rs if r[3] == 1]
    fl = sum(r[9] != orig[r[2]][9] for r in rs if r[2] in orig)
    d1 = st.mean(abs(r[5] - orig[r[2]][5]) for r in rs if r[2] in orig); d3 = st.mean(abs(r[7] - orig[r[2]][7]) for r in rs if r[2] in orig)
    print(f"  {vn:18s} | {len(rs):3d} | {sum(1-r[9] for r in re_):3d}/{len(re_)} | {sum(r[9] for r in fa):3d}/{len(fa)} | "
          f"{st.mean(r[5] for r in rs):.3f} | {st.mean(r[6] for r in rs):.3f} | {st.mean(r[7] for r in rs):.3f} | {fl:3d} | {d1:.3f} | {d3:.3f}")
print(f"\nSaved {OUT}/robustness_audio_scores.csv")
shutil.rmtree(TMP, ignore_errors=True)
