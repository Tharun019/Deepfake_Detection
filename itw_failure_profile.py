"""
What distinguishes the real (bona-fide) In-the-Wild clips the model calls FAKE?
Reads itw_zero_shot_results.csv (no model needed), measures simple acoustic
properties of every scored bona-fide clip, and compares misclassified vs
correctly-classified ones. Also profiles any clips you pass on the command line.
Correlational diagnostic only - it suggests augmentation targets, proves no cause.

Usage: python itw_failure_profile.py [your_clip1.mp3 ...]
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

from modules.content.audio_utils import preprocess_audio, TARGET_SAMPLE_RATE

SR = TARGET_SAMPLE_RATE
ITW_DIR = os.environ.get('ITW_DIR', os.path.expanduser('~/Desktop/Deepfake_Detection/datasets/itw/release_in_the_wild'))
RES_CSV = os.path.join(os.path.dirname(os.path.abspath(ITW_DIR)), 'itw_zero_shot_results.csv')
FRAME = 320


def frame_db(x):
    n = len(x) // FRAME
    rms = np.sqrt((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms)


def features(x):
    f = {'duration_s': len(x) / SR}
    db = frame_db(x)
    f['loudness_db'] = float(20 * np.log10(np.sqrt((x ** 2).mean() + 1e-12)))
    f['noise_floor_db'] = float(np.percentile(db, 10))
    f['dynamic_range_db'] = float(np.percentile(db, 95) - np.percentile(db, 10))
    nfft, hop = 512, 256
    if len(x) >= nfft:
        n = (len(x) - nfft) // hop + 1
        idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
        ps = (np.abs(np.fft.rfft(x[idx] * np.hanning(nfft)[None, :], axis=1)) ** 2).mean(axis=0)
        freqs = np.fft.rfftfreq(nfft, 1 / SR)
        tot = ps.sum() + 1e-12
        f['hf_ratio_4k_8k'] = float(ps[freqs >= 4000].sum() / tot)
        f['rolloff99_khz'] = float(freqs[min(np.searchsorted(np.cumsum(ps) / tot, 0.99), len(freqs) - 1)] / 1000)
    else:
        f['hf_ratio_4k_8k'] = f['rolloff99_khz'] = float('nan')
    f['clipping_frac'] = float((np.abs(x) > 0.99).mean())
    thr = (np.sqrt((x[:(len(x) // FRAME) * FRAME].reshape(-1, FRAME) ** 2).mean(axis=1) + 1e-12)).max() * 10 ** (-35 / 20)
    act = np.where(np.sqrt((x[:(len(x) // FRAME) * FRAME].reshape(-1, FRAME) ** 2).mean(axis=1) + 1e-12) > thr)[0]
    f['lead_silence_s'] = float(act[0] * FRAME / SR) if len(act) else 0.0
    return f


def auc(pos, neg):
    """P(random pos value > random neg value) via ranks (ties averaged)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind='mergesort')
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                      # average tied ranks
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


rows = [r for r in csv.DictReader(open(RES_CSV)) if r['label'] == 'bona-fide' and r['raw_fake_prob'] != '']
for r in rows:
    r['p'] = float(r['raw_fake_prob'])
fail = [r for r in rows if r['p'] >= 0.5]
print(f'bona-fide clips scored: {len(rows)}   misclassified as fake (p>=0.5): {len(fail)} ({100 * len(fail) / len(rows):.1f}%)')

# ---- per-speaker concentration ----
by_spk = defaultdict(list)
for r in rows:
    by_spk[r['speaker']].append(r['p'])
spk = [(s, len(v), float(np.mean(np.array(v) >= 0.5)), float(np.median(v))) for s, v in by_spk.items()]
big = sorted([t for t in spk if t[1] >= 10], key=lambda t: -t[2])
print(f'\nSpeakers with bona-fide clips: {len(spk)}   (showing those with >=10 clips)')
print('Highest misclassification rate:')
for s, n, fr, med in big[:8]:
    print(f'   {s[:28]:28s} n={n:3d}   called fake: {100 * fr:5.1f}%   median fake prob={med:.3f}')
print('Lowest misclassification rate:')
for s, n, fr, med in big[-4:]:
    print(f'   {s[:28]:28s} n={n:3d}   called fake: {100 * fr:5.1f}%   median fake prob={med:.3f}')
fail_by_spk = sorted(((sum(1 for p in v if p >= 0.5), s) for s, v in by_spk.items()), reverse=True)
top5 = sum(c for c, _ in fail_by_spk[:5])
print(f'Top 5 speakers account for {top5} of {len(fail)} misclassified clips ({100 * top5 / max(len(fail), 1):.0f}%)')

# ---- acoustic features: misclassified vs correct ----
feats = []
for i, r in enumerate(rows, 1):
    x = preprocess_audio(os.path.join(ITW_DIR, r['file'])).numpy().astype(np.float32)
    feats.append(features(x))
    if i % 200 == 0:
        print(f'  measured {i}/{len(rows)}', flush=True)
names = list(feats[0].keys())
isfail = np.array([r['p'] >= 0.5 for r in rows])
print('\n=========== misclassified-real vs correctly-classified-real (bona-fide only) ===========')
print(f'{"feature":20s} {"median(correct)":>16s} {"median(misclass.)":>18s} {"AUC":>6s}   (AUC>0.5: higher value => more likely misclassified)')
table = []
for k in names:
    v = np.array([f[k] for f in feats], float)
    ok = ~np.isnan(v)
    a = auc(v[ok & isfail], v[ok & ~isfail])
    table.append((abs(a - 0.5), k, np.median(v[ok & ~isfail]), np.median(v[ok & isfail]), a))
for _, k, mc, mf, a in sorted(table, reverse=True):
    print(f'{k:20s} {mc:16.3f} {mf:18.3f} {a:6.2f}')

# ---- your clips ----
if len(sys.argv) > 1:
    print('\n=========== YOUR CLIPS (same measurements) ===========')
    for path in sys.argv[1:]:
        try:
            f = features(preprocess_audio(path).numpy().astype(np.float32))
            print(os.path.basename(path))
            print('   ' + '   '.join(f'{k}={v:.3f}' for k, v in f.items()))
        except Exception as e:
            print(path, 'ERROR', e)
