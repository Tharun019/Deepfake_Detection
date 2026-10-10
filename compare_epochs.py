"""
Compare the ORIGINAL L2 audio checkpoint against every fine-tuned epoch, on exactly the
same held-out data, and print everything needed to choose an epoch.

Models scored : baseline (checkpoints/wav2vec2_asvspoof_augmented_v2/epoch_2) + each epoch_N
                found in the fine-tune run folder.
Data          : ITW validation speakers (selection), ITW test speakers (unseen), an ASVspoof
                dev sample (regression), and any extra clips you pass in.
Method        : deployed inference (4 s windows over the whole clip, median). Sets are rebuilt
                from split.json with the same seed/caps as train_itw_finetune.py, so the baseline
                must reproduce the earlier BEFORE numbers (checked below).
Nothing in the project is modified. Per-clip scores are cached next to the checkpoints, so a
re-run only re-scores the clips.

Usage:
  python compare_epochs.py --real_clips CLIP1.mp3 CLIP2.mp3 [--fake_clips FILE_OR_DIR ...] [--denoise]
  (--real_clips / --fake_clips accept files or folders)
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from modules.content.audio_utils import preprocess_audio, TARGET_SAMPLE_RATE
from train_itw_finetune import window_probs, auc_eer, cap_per_class
from train_wav2vec2 import parse_protocol_file, get_device, LABEL_FAKE, LABEL_REAL

SR = TARGET_SAMPLE_RATE
AUDIO_EXT = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.opus'}
SEL_ASV_TOL = 0.03        # an epoch is ineligible if ASVspoof-dev balanced accuracy drops > 3 pts vs baseline
SET_KEYS = ('val', 'test', 'asv')


# ------------------------------------------------------------------ helpers
def collect(paths):
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for dp, _, fns in os.walk(p):
                for fn in sorted(fns):
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXT and not fn.startswith('.'):
                        out.append(os.path.join(dp, fn))
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f'[warn] not found, skipped: {p}')
    return out


def score_items(model, fe, device, items, tag):
    """items: [(path, label)] with LABEL_REAL/LABEL_FAKE. Returns [(path, y_fake, median_fake_prob)]."""
    out, fails, t0 = [], 0, time.time()
    for i, (path, label) in enumerate(items, 1):
        try:
            p = float(np.median(window_probs(model, fe, device, preprocess_audio(path))))
            out.append((path, int(label == LABEL_FAKE), p))
        except Exception:
            fails += 1
        if i % 500 == 0:
            print(f'     {tag}: {i}/{len(items)} ({time.time() - t0:.0f}s)', flush=True)
    if fails:
        print(f'     [warn] {tag}: {fails} clips failed to load/score')
    return out


def score_clip(model, fe, device, path, denoise=False, nr=None):
    w = preprocess_audio(path)
    if denoise:
        w = torch.from_numpy(np.asarray(nr.reduce_noise(y=w.numpy(), sr=SR), dtype=np.float32))
    return window_probs(model, fe, device, w)


def metrics(sc, thr=0.5):
    y = np.array([s[1] for s in sc]); p = np.array([s[2] for s in sc])
    real, fake = p[y == 0], p[y == 1]
    m = {'n_real': int(len(real)), 'n_fake': int(len(fake))}
    if len(real) == 0 or len(fake) == 0:
        return m
    m['rr'] = float((real < thr).mean()); m['fr'] = float((fake >= thr).mean())
    m['ba'] = (m['rr'] + m['fr']) / 2
    m['auc'], m['eer'] = auc_eer(y, p)
    return m


def bal_at(sc, thr):
    y = np.array([s[1] for s in sc]); p = np.array([s[2] for s in sc])
    return float(((p[y == 0] < thr).mean() + (p[y == 1] >= thr).mean()) / 2)


def best_thr(sc):
    """Threshold maximising balanced accuracy on `sc` (ties -> closest to 0.5). Chosen on VAL only."""
    best = None
    for t in np.round(np.arange(0.02, 0.981, 0.01), 2):
        key = (bal_at(sc, float(t)), -abs(float(t) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def spk_counts(sc, path2spk, spks):
    """per-speaker [n_real, real_correct, n_fake, fake_correct] at 0.5, rows ordered like spks."""
    idx = {s: i for i, s in enumerate(spks)}
    c = np.zeros((len(spks), 4))
    for path, y, p in sc:
        r = c[idx[path2spk[path]]]
        if y == 0:
            r[0] += 1; r[1] += (p < 0.5)
        else:
            r[2] += 1; r[3] += (p >= 0.5)
    return c


def bootstrap(counts, base, B, seed=0):
    """Speaker-cluster bootstrap. counts[name] is (n_speakers, 4). Returns per-model stats."""
    names = list(counts)
    S = counts[base].shape[0]
    rng = np.random.default_rng(seed)
    ba = {n: [] for n in names}
    for _ in range(B):
        idx = rng.integers(0, S, S)
        tot = {n: counts[n][idx].sum(axis=0) for n in names}
        if tot[base][0] == 0 or tot[base][2] == 0:
            continue
        for n in names:
            ba[n].append(0.5 * (tot[n][1] / tot[n][0] + tot[n][3] / tot[n][2]))
    out = {}
    for n in names:
        a = np.array(ba[n]); d = a - np.array(ba[base])
        out[n] = {'lo': float(np.percentile(a, 2.5)), 'hi': float(np.percentile(a, 97.5)),
                  'dlo': float(np.percentile(d, 2.5)), 'dhi': float(np.percentile(d, 97.5)),
                  'p_better': float((d > 0).mean()), 'draws': len(a)}
    return out


def f1(x):
    return f'{100 * x:5.1f}'


# ------------------------------------------------------------------ report
def make_report(models, S, path2spk, clips, CR, modes, B):
    base = models[0]
    epochs = models[1:]
    M = {n: {k: metrics(S[n][k]) for k in SET_KEYS} for n in models}
    thr = {n: best_thr(S[n]['val']) for n in models}

    print('\n' + '=' * 100)
    print('SANITY CHECK - baseline must reproduce the numbers from the earlier BEFORE scoring')
    t, a = M[base]['test'], M[base]['asv']
    ok = (abs(100 * t['rr'] - 56.7) <= 0.15 and abs(100 * t['fr'] - 76.1) <= 0.15
          and abs(100 * a['rr'] - 100.0) <= 0.15 and abs(100 * a['fr'] - 88.0) <= 0.15)
    print(f'   baseline ITW-test real {100 * t["rr"]:.1f}% / spoof {100 * t["fr"]:.1f}%  (earlier: 56.7 / 76.1)')
    print(f'   baseline ASV-dev  real {100 * a["rr"]:.1f}% / spoof {100 * a["fr"]:.1f}%  (earlier: 100.0 / 88.0)')
    print('   ->', 'REPRODUCED - the sets are identical to the training run' if ok else
          'NOT REPRODUCED - the sets differ from the training run, do not trust the comparison below')

    print('\n' + '=' * 100)
    print('TABLE 1 - ITW VALIDATION speakers (the only data used to choose an epoch)')
    print(f'{"model":10s} {"real->REAL":>10s} {"spoof->FAKE":>11s} {"bal.acc":>8s} {"AUC":>6s} {"EER":>6s} | {"best thr":>8s} {"bal.acc@thr":>11s}')
    for n in models:
        m = M[n]['val']
        print(f'{n:10s} {f1(m["rr"]):>10s} {f1(m["fr"]):>11s} {f1(m["ba"]):>8s} {m["auc"]:6.3f} {f1(m["eer"]):>6s} | '
              f'{thr[n]:8.2f} {f1(bal_at(S[n]["val"], thr[n])):>11s}')

    print('\nTABLE 2 - ITW TEST speakers (never seen in training; reported, not used for selection)')
    print(f'{"model":10s} {"real->REAL":>10s} {"spoof->FAKE":>11s} {"bal.acc":>8s} {"AUC":>6s} {"EER":>6s} | {"bal.acc@val-thr":>15s}')
    for n in models:
        m = M[n]['test']
        print(f'{n:10s} {f1(m["rr"]):>10s} {f1(m["fr"]):>11s} {f1(m["ba"]):>8s} {m["auc"]:6.3f} {f1(m["eer"]):>6s} | '
              f'{f1(bal_at(S[n]["test"], thr[n])):>15s}')
    print('   (bal.acc@val-thr = threshold picked on VAL only, then applied to TEST - shows whether a gap is just calibration)')

    print('\nTABLE 3 - ASVspoof DEV sample (regression check: has it forgotten studio audio?)')
    print(f'{"model":10s} {"real->REAL":>10s} {"spoof->FAKE":>11s} {"bal.acc":>8s} {"AUC":>6s}   guard (bal.acc >= baseline - {100 * SEL_ASV_TOL:.0f} pts)')
    eligible = []
    for n in models:
        m = M[n]['asv']
        if n == base:
            g = 'baseline'
        else:
            okg = m['ba'] >= M[base]['asv']['ba'] - SEL_ASV_TOL
            g = 'ok' if okg else 'FAIL'
            if okg:
                eligible.append(n)
        print(f'{n:10s} {f1(m["rr"]):>10s} {f1(m["fr"]):>11s} {f1(m["ba"]):>8s} {m["auc"]:6.3f}   {g}')

    print('\n' + '=' * 100)
    print('EPOCH SELECTION')
    print('RULE (fixed before this comparison; same criterion as the training script, plus the ASV guard):')
    print('   among epochs that pass the ASV guard, take the highest ITW-val balanced accuracy at 0.5 (tie: higher val AUC).')
    if eligible:
        sel = max(eligible, key=lambda n: (M[n]['val']['ba'], M[n]['val']['auc']))
        print(f'   >>> SELECTED BY RULE: {sel}')
    else:
        sel = None
        print('   >>> no epoch passes the guard')
    print('Sensitivity (reported only - NOT used to select):')
    print(f'   highest val AUC      : {max(epochs, key=lambda n: M[n]["val"]["auc"])}')
    print(f'   lowest  val EER      : {min(epochs, key=lambda n: M[n]["val"]["eer"])}')
    print(f'   highest val bal.acc at its own best threshold: {max(epochs, key=lambda n: bal_at(S[n]["val"], thr[n]))}')
    print(f'   highest TEST bal.acc @0.5 (look, do not pick): {max(epochs, key=lambda n: M[n]["test"]["ba"])}')

    print('\n' + '=' * 100)
    spks = sorted({path2spk[s[0]] for s in S[base]['test']})
    counts = {n: spk_counts(S[n]['test'], path2spk, spks) for n in models}
    bs = bootstrap(counts, base, B)
    print(f'UNCERTAINTY - speaker-cluster bootstrap on TEST ({len(spks)} speakers resampled, {bs[base]["draws"]} draws)')
    print('   Clips from one speaker are correlated, so resampling speakers (not clips) gives the honest interval.')
    print(f'{"model":10s} {"bal.acc":>8s} {"95% interval":>16s} | {"change vs baseline":>18s} {"95% interval":>18s} {"share of draws better":>22s}')
    for n in models:
        m = M[n]['test']; b = bs[n]
        ci = '[%.1f, %.1f]' % (100 * b['lo'], 100 * b['hi'])
        if n == base:
            print('%-10s %8s %16s |' % (n, f1(m['ba']), ci))
        else:
            dci = '[%+.1f, %+.1f]' % (100 * b['dlo'], 100 * b['dhi'])
            dlt = '%+.1f pts' % (100 * (m['ba'] - M[base]['test']['ba']))
            share = '%.0f%%' % (100 * b['p_better'])
            print('%-10s %8s %16s | %18s %18s %22s' % (n, f1(m['ba']), ci, dlt, dci, share))

    print('\nPER-SPEAKER on TEST  (cells = real->REAL% / spoof->FAKE%, "-" = no clips of that class in the sample)')
    print(f'{"speaker":24s} {"n real+fake":>12s} ' + ' '.join(f'{n:>9s}' for n in models))
    for i, s in enumerate(spks):
        c0 = counts[base][i]
        cells = []
        for n in models:
            c = counts[n][i]
            r = f'{100 * c[1] / c[0]:3.0f}' if c[0] else '  -'
            f = f'{100 * c[3] / c[2]:3.0f}' if c[2] else '  -'
            cells.append(f'{r}/{f}'.rjust(9))
        print(f'{s[:24]:24s} {int(c0[0]):5d}+{int(c0[2]):<5d} ' + ' '.join(cells))

    if clips:
        for mode in modes:
            print('\n' + '=' * 100)
            print('YOUR CLIPS - ' + ('plain (current pipeline)' if mode == 'plain' else
                  'WITH noisereduce DENOISING applied before windowing (quick check on these clips only, NOT a validation)'))
            print('   cell = median window fake-prob (windows >= 50% fake / total windows); verdict at 0.5 on the median')
            print(f'{"clip":42s} {"truth":5s} ' + ' '.join(f'{n:>16s}' for n in models))
            for c in clips:
                cells = []
                for n in models:
                    wp = CR[n][mode].get(c['path'])
                    if wp is None:
                        cells.append('error'.rjust(16))
                    else:
                        cells.append(f'{100 * np.median(wp):5.1f}% {int((wp >= 0.5).sum()):2d}/{len(wp):<2d}'.rjust(16))
                print(f'{os.path.basename(c["path"])[:42]:42s} {c["label"]:5s} ' + ' '.join(cells))
            for n in models:
                rc = [c for c in clips if c['label'] == 'real']; fc = [c for c in clips if c['label'] == 'fake']
                def called_fake(c):
                    wp = CR[n][mode].get(c['path'])
                    return None if wp is None else bool(np.median(wp) >= 0.5)
                rr = sum(1 for c in rc if called_fake(c) is False)
                ff = sum(1 for c in fc if called_fake(c) is True)
                print(f'   {n:10s} real clips called REAL: {rr}/{len(rc)}   fake clips called FAKE: {ff}/{len(fc)}')
            if mode == 'plain' and len(clips) <= 4:
                print('\n   per-window fake probability (%):')
                for c in clips:
                    print(f'   {os.path.basename(c["path"])[:60]}')
                    for n in models:
                        wp = CR[n][mode].get(c['path'])
                        print(f'      {n:10s} ' + ('error' if wp is None else str([round(100 * float(v), 1) for v in wp])))
    print('\n' + '=' * 100)
    print('Paste everything above this line.')


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--itw_dir', default=os.path.expanduser('~/Desktop/Deepfake_Detection/datasets/itw/release_in_the_wild'))
    ap.add_argument('--asv_root', default=os.path.expanduser('~/.cache/kagglehub/datasets/anishsarkar22/asvpoof-2019-dataset-la/versions/1'))
    ap.add_argument('--init_checkpoint', default=str(ROOT / 'checkpoints/wav2vec2_asvspoof_augmented_v2/epoch_2'))
    ap.add_argument('--run_dir', default=str(ROOT / 'checkpoints/wav2vec2_itw_ft_v1'))
    ap.add_argument('--real_clips', nargs='*', default=[])
    ap.add_argument('--fake_clips', nargs='*', default=[])
    ap.add_argument('--denoise', action='store_true', help='also score the clips after noisereduce denoising')
    ap.add_argument('--val_per_class', type=int, default=500)
    ap.add_argument('--test_per_class', type=int, default=1500)
    ap.add_argument('--asv_dev_per_class', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--no_cache', action='store_true')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    split_path = run_dir / 'split.json'
    if not split_path.exists():
        sys.exit(f'{split_path} not found - run train_itw_finetune.py first')
    epoch_dirs = sorted([d for d in run_dir.glob('epoch_*') if (d / 'config.json').exists()],
                        key=lambda d: int(d.name.split('_')[1]))
    if not epoch_dirs:
        sys.exit(f'No epoch_* checkpoints in {run_dir}')
    models = {'baseline': Path(args.init_checkpoint)}
    models.update({d.name: d for d in epoch_dirs})
    names = list(models)

    nr = None
    if args.denoise:
        try:
            import noisereduce as nr
        except ImportError:
            print('[warn] noisereduce is not installed -> run: pip install noisereduce   (denoise comparison skipped)')
            args.denoise = False
    modes = ['plain'] + (['denoise'] if args.denoise else [])

    # ---- rebuild the exact held-out sets
    split = json.load(open(split_path))
    spk2split = {s: k for k in ('train', 'val', 'test') for s in split[k]}
    rows = list(csv.DictReader(open(os.path.join(args.itw_dir, 'meta.csv'))))
    items = {'train': [], 'val': [], 'test': []}
    path2spk = {}
    for r in rows:
        p = os.path.join(args.itw_dir, r['file'])
        if os.path.exists(p):
            items[spk2split[r['speaker']]].append((p, LABEL_REAL if r['label'] == 'bona-fide' else LABEL_FAKE))
            path2spk[p] = r['speaker']
    sets = {'val': cap_per_class(items['val'], args.val_per_class, args.seed),
            'test': cap_per_class(items['test'], args.test_per_class, args.seed)}
    la = os.path.join(args.asv_root, 'LA')
    dev_labels = parse_protocol_file(os.path.join(la, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt'))
    dev_flac = os.path.join(la, 'ASVspoof2019_LA_dev', 'flac')
    dev_items = [(os.path.join(dev_flac, f'{n}.flac'), l) for n, l in dev_labels.items()
                 if os.path.exists(os.path.join(dev_flac, f'{n}.flac'))]
    sets['asv'] = cap_per_class(dev_items, args.asv_dev_per_class, args.seed)
    print(f'[sets] ITW val {len(sets["val"])} | ITW test {len(sets["test"])} ({len({path2spk[p] for p, _ in sets["test"]})} speakers) | ASV dev {len(sets["asv"])}')

    clips = [{'path': p, 'label': 'real'} for p in collect(args.real_clips)] + \
            [{'path': p, 'label': 'fake'} for p in collect(args.fake_clips)]
    print(f'[clips] {sum(c["label"] == "real" for c in clips)} real, {sum(c["label"] == "fake" for c in clips)} fake'
          + (' | denoise comparison ON' if args.denoise else ''))

    device = get_device()
    S, CR = {}, {}
    for name in names:
        print(f'\n[{name}] {models[name]}', flush=True)
        cache_file = run_dir / f'compare_cache_{name}.json'
        cache = {}
        if cache_file.exists() and not args.no_cache:
            try:
                cache = json.load(open(cache_file))
            except Exception:
                cache = {}
        fe = Wav2Vec2FeatureExtractor.from_pretrained(models[name])
        model = Wav2Vec2ForSequenceClassification.from_pretrained(models[name]).to(device)
        model.eval()
        S[name] = {}
        for k in SET_KEYS:
            want = {p for p, _ in sets[k]}
            if k in cache and {c[0] for c in cache[k]} == want:
                S[name][k] = [tuple(c) for c in cache[k]]
                print(f'   {k}: loaded {len(S[name][k])} cached scores')
            else:
                t0 = time.time()
                S[name][k] = score_items(model, fe, device, sets[k], f'{name}/{k}')
                cache[k] = [list(c) for c in S[name][k]]
                print(f'   {k}: scored {len(S[name][k])} clips in {time.time() - t0:.0f}s', flush=True)
        json.dump(cache, open(cache_file, 'w'))
        CR[name] = {m: {} for m in modes}
        for c in clips:
            for m in modes:
                try:
                    CR[name][m][c['path']] = score_clip(model, fe, device, c['path'], denoise=(m == 'denoise'), nr=nr)
                except Exception as e:
                    print(f'   [warn] {os.path.basename(c["path"])} ({m}): {e}')
                    CR[name][m][c['path']] = None
        del model
        try:
            if device.type == 'mps':
                torch.mps.empty_cache()
        except Exception:
            pass

    make_report(names, S, path2spk, clips, CR, modes, args.boot)


if __name__ == '__main__':
    main()
