"""
Silence-shortcut diagnostic for the L2 audio model (Wav2Vec2 fine-tuned on
ASVspoof 2019 LA). Question: does the model rely on leading/trailing silence
(a known ASVspoof 2019 dataset artifact) rather than on speech content?

Runs the CURRENT checkpoint on a seeded random sample of ASVspoof dev clips
under several conditions, and measures silence on your own real clips.
Does not modify any project file.

Usage: python asv_silence_test.py [your_clip1.mp3 your_clip2.mp3 ...]
Env:   N_PER_CLASS (default 200), SILENCE_DB (default -35), ASV_ROOT
"""
import os
import sys
import random

import numpy as np
import torch

from modules.content.audio_analyzer import _get_audio_model, MAX_AUDIO_SAMPLES
from modules.content.audio_utils import preprocess_audio, TARGET_SAMPLE_RATE

SR = TARGET_SAMPLE_RATE
WIN = MAX_AUDIO_SAMPLES                    # 4 s, what the model was trained on
FRAME = 320                                # 20 ms
MIN_SAMPLES = 4000                         # 0.25 s; shorter inputs are skipped
MIN_TRIMMED = 8000                         # if trimming leaves < 0.5 s, keep original
N_PER_CLASS = int(os.environ.get('N_PER_CLASS', '200'))
SILENCE_DB = float(os.environ.get('SILENCE_DB', '-35'))
SEED = 42
ROOT = os.environ.get(
    'ASV_ROOT',
    os.path.expanduser('~/.cache/kagglehub/datasets/anishsarkar22/asvpoof-2019-dataset-la/versions/1'),
)

model, fe, dev = _get_audio_model()


def fake_prob(x):
    inputs = fe(x, sampling_rate=SR, return_tensors='pt')
    iv = inputs['input_values'].to(dev)
    with torch.no_grad():
        logits = model(input_values=iv).logits
    return float(torch.softmax(logits, dim=-1)[0, 0])      # index 0 = fake


def prod_window(x):
    """Exactly what production does: first 4 s, zero-padded at the end if shorter."""
    w = x[:WIN]
    if w.shape[0] < WIN:
        w = np.pad(w, (0, WIN - w.shape[0]))
    return w


def silence_bounds(x):
    """(start, end) sample indices of the non-silent region; silence = frames more
    than SILENCE_DB below the loudest 20 ms frame."""
    n = len(x) // FRAME
    if n == 0:
        return 0, len(x)
    rms = np.sqrt((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1) + 1e-12)
    thr = rms.max() * 10 ** (SILENCE_DB / 20)
    active = np.where(rms > thr)[0]
    if len(active) == 0:
        return 0, len(x)
    return int(active[0] * FRAME), int((active[-1] + 1) * FRAME)


def trim(x):
    s, e = silence_bounds(x)
    t = x[s:e]
    if t.shape[0] < MIN_TRIMMED:
        t = x
    return t, s / SR, (len(x) - e) / SR


def safe_prob(x):
    return fake_prob(x) if x.shape[0] >= MIN_SAMPLES else None


# ---------- locate ASVspoof dev protocol + audio ----------
protocols, flac_index = [], {}
for dp, _, fns in os.walk(ROOT):
    for fn in fns:
        low = fn.lower()
        if low.endswith('.txt') and 'dev' in low and 'cm' in low:
            protocols.append(os.path.join(dp, fn))
        elif low.endswith('.flac'):
            flac_index[fn[:-5]] = os.path.join(dp, fn)
protocols.sort(key=lambda p: ('trl' not in p.lower(), p))
print(f'Dataset root : {ROOT}')
print(f'Protocol file candidates: {protocols}')
print(f'Indexed .flac files: {len(flac_index)}')
if not protocols or not flac_index:
    sys.exit('Could not find dev protocol / flac files under ASV_ROOT. Paste this output back to me.')
proto = protocols[0]
print(f'Using protocol: {proto}\n')

pool = {'bonafide': [], 'spoof': []}
with open(proto) as f:
    for line in f:
        p = line.split()
        if len(p) >= 5 and p[-1] in pool and p[1] in flac_index:
            pool[p[-1]].append(p[1])
rng = random.Random(SEED)
sample = {k: rng.sample(sorted(v), min(N_PER_CLASS, len(v))) for k, v in pool.items()}
print({k: f'{len(v)} sampled of {len(pool[k])} available' for k, v in sample.items()}, '\n')

# ---------- run conditions ----------
res = {k: {'A_production': [], 'B_unpadded': [], 'C_trimmed_unpadded': [], 'D_plus1s_leading_silence': []}
       for k in sample}
sil = {k: {'dur': [], 'lead': [], 'trail': []} for k in sample}

for label, names in sample.items():
    for i, name in enumerate(names, 1):
        x = preprocess_audio(flac_index[name]).numpy().astype(np.float32)
        t, lead, trail = trim(x)
        sil[label]['dur'].append(len(x) / SR)
        sil[label]['lead'].append(lead)
        sil[label]['trail'].append(trail)
        res[label]['A_production'].append(fake_prob(prod_window(x)))
        for key, arr in (('B_unpadded', x[:WIN]), ('C_trimmed_unpadded', t[:WIN])):
            v = safe_prob(arr)
            if v is not None:
                res[label][key].append(v)
        if label == 'spoof':
            res[label]['D_plus1s_leading_silence'].append(
                fake_prob(prod_window(np.concatenate([np.zeros(SR, np.float32), x]))))
        if i % 25 == 0:
            print(f'  {label}: {i}/{len(names)}', flush=True)

print('\n=========== RESULTS (ASVspoof 2019 LA dev sample, current checkpoint) ===========')
print(f'Silence threshold: {SILENCE_DB} dB below loudest 20 ms frame\n')
for label in ('bonafide', 'spoof'):
    s = sil[label]
    print(f'[{label}]  n={len(s["dur"])}  median duration={np.median(s["dur"]):.2f}s  '
          f'median leading silence={np.median(s["lead"]):.2f}s  '
          f'median trailing silence={np.median(s["trail"]):.2f}s  '
          f'clips with leading silence >=0.5s: {100 * np.mean(np.array(s["lead"]) >= 0.5):.0f}%')
    for cond, probs in res[label].items():
        if not probs:
            continue
        p = np.array(probs)
        correct = (p < 0.5).mean() if label == 'bonafide' else (p >= 0.5).mean()
        print(f'   {cond:26s} correct={100 * correct:5.1f}%   mean fake prob={p.mean():.3f}   n={len(p)}')
    print()
print('(correct = classified REAL for bonafide, FAKE for spoof, threshold 0.5)')

# ---------- your own clips ----------
if len(sys.argv) > 1:
    print('\n=========== YOUR CLIPS ===========')
    for path in sys.argv[1:]:
        try:
            x = preprocess_audio(path).numpy().astype(np.float32)
            t, lead, trail = trim(x)
            a = fake_prob(prod_window(x))
            b = safe_prob(x[:WIN])
            c = safe_prob(t[:WIN])
            print(os.path.basename(path))
            print(f'   duration={len(x) / SR:.2f}s  leading silence={lead:.2f}s  trailing silence={trail:.2f}s')
            print(f'   first-4s fake prob: production={a:.3f}  unpadded={b:.3f}  after trimming={c:.3f}')
        except Exception as e:
            print(path, 'ERROR', e)
