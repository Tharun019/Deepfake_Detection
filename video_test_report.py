import csv
from collections import defaultdict
rows = list(csv.DictReader(open("results/video_test_eval.csv")))
ok = [r for r in rows if r["raw_fake_prob"] != "" and not r["error"]]
print(f"rows: {len(rows)} | scored: {len(ok)} | failed: {len(rows)-len(ok)}")
real = [r for r in ok if r["label"] == "1"]
fake = [r for r in ok if r["label"] == "0"]
rc = sum(float(r["raw_fake_prob"]) < 0.5 for r in real)
fc = sum(float(r["raw_fake_prob"]) >= 0.5 for r in fake)
rr, fr = rc / len(real), fc / len(fake)
print(f"real recall: {rc}/{len(real)} = {rr*100:.1f}%")
print(f"fake recall: {fc}/{len(fake)} = {fr*100:.1f}%")
print(f"BALANCED ACCURACY: {(rr+fr)/2*100:.1f}%   (claimed: 70.1%)")
try:
    from sklearn.metrics import roc_auc_score
    y = [1 - int(r["label"]) for r in ok]; p = [float(r["raw_fake_prob"]) for r in ok]
    print(f"AUC: {roc_auc_score(y, p):.3f}")
except Exception as ex:
    print("AUC skipped:", ex)
by = defaultdict(list)
for r in fake: by[r["method"]].append(float(r["raw_fake_prob"]) >= 0.5)
for m, v in sorted(by.items()):
    print(f"  {m:15} fake recall {sum(v)}/{len(v)} = {sum(v)/len(v)*100:.1f}%")
