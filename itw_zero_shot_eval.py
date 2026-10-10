"""
Zero-shot evaluation of the CURRENT deployed L2 audio pipeline (windowed Wav2Vec2,
median aggregation) on the In-the-Wild dataset (Mueller et al., 2022).
No training, no project file modified. Writes per-file results to a CSV.

Usage: python itw_zero_shot_eval.py
Env:   N_PER_CLASS (default 1000), ITW_DIR
"""
import csv
import os
import random
import sys
import time

import numpy as np

from modules.content.audio_analyzer import analyze_audio

ITW_DIR = os.environ.get('ITW_DIR', os.path.expanduser('~/Desktop/Deepfake_Detection/datasets/itw/release_in_the_wild'))
N_PER_CLASS = int(os.environ.get('N_PER_CLASS', '1000'))
SEED = 42
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(ITW_DIR)), 'itw_zero_shot_results.csv')

rows = list(csv.DictReader(open(os.path.join(ITW_DIR, 'meta.csv'))))
labels = sorted({r['label'] for r in rows})
print(f'meta.csv rows: {len(rows)}   labels found: {labels}   speakers: {len({r["speaker"] for r in rows})}')
if set(labels) != {'bona-fide', 'spoof'}:
    sys.exit(f'Unexpected label values {labels} - paste this output back to me.')

by_label = {'bona-fide': [], 'spoof': []}
for r in rows:
    by_label[r['label']].append(r)
print({k: len(v) for k, v in by_label.items()}, '(clips per class in full dataset)')

rng = random.Random(SEED)
sample = {k: rng.sample(v, min(N_PER_CLASS, len(v))) for k, v in by_label.items()}

results = []   # (file, speaker, label, y_fake, score, duration, error)
t0 = time.time()
for label, items in sample.items():
    for i, r in enumerate(items, 1):
        out = analyze_audio(os.path.join(ITW_DIR, r['file']))
        f = out['features']
        if 'error' in f:
            results.append((r['file'], r['speaker'], label, int(label == 'spoof'), None, None, f['error']))
        else:
            results.append((r['file'], r['speaker'], label, int(label == 'spoof'),
                            f['raw_fake_prob'], f['original_duration_sec'], ''))
        if i % 100 == 0:
            print(f'  {label}: {i}/{len(items)}   ({time.time() - t0:.0f}s elapsed)', flush=True)

with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['file', 'speaker', 'label', 'y_fake', 'raw_fake_prob', 'duration_sec', 'error'])
    w.writerows(results)

ok = [r for r in results if r[4] is not None]
errs = len(results) - len(ok)
y = np.array([r[3] for r in ok])
s = np.array([r[4] for r in ok])


def auc_eer(y, s):
    order = np.argsort(-s, kind='mergesort')
    ys = y[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    tpr, fpr = np.r_[0, tps / tps[-1]], np.r_[0, fps / fps[-1]]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return auc, float((fnr[i] + fpr[i]) / 2)


auc, eer = auc_eer(y, s)
real, fake = s[y == 0], s[y == 1]
real_rec, fake_rec = float((real < 0.5).mean()), float((fake >= 0.5).mean())

print('\n=========== In-the-Wild ZERO-SHOT (current deployed L2 audio pipeline) ===========')
print(f'clips scored: {len(ok)}  (load/analysis errors: {errs})   seed={SEED}   median aggregation')
print(f'bona-fide  n={len(real)}   classified REAL at 0.5: {100 * real_rec:.1f}%   mean fake prob={real.mean():.3f}   median={np.median(real):.3f}')
print(f'spoof      n={len(fake)}   classified FAKE at 0.5: {100 * fake_rec:.1f}%   mean fake prob={fake.mean():.3f}   median={np.median(fake):.3f}')
print(f'balanced accuracy @0.5: {100 * (real_rec + fake_rec) / 2:.1f}%')
print(f'AUC: {auc:.3f}    EER: {100 * eer:.1f}%   (AUC 0.5 / EER 50% = chance)')
print('bona-fide fake-prob percentiles (10/25/50/75/90): ' + ' / '.join(f'{np.percentile(real, q):.3f}' for q in (10, 25, 50, 75, 90)))
print('spoof     fake-prob percentiles (10/25/50/75/90): ' + ' / '.join(f'{np.percentile(fake, q):.3f}' for q in (10, 25, 50, 75, 90)))
print(f'\nPer-file results saved to: {OUT_CSV}')
