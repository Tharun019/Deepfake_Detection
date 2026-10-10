"""
Fine-tune the deployed L2 audio model on In-the-Wild (ITW) using a speaker-disjoint
split, mixed with a sample of ASVspoof 2019 LA train so it keeps its studio-audio
skill. Starts from the CURRENT checkpoint and never overwrites it.

Model selection uses ITW *validation* speakers only. ITW *test* speakers, an
ASVspoof dev sample and your own clips are scored BEFORE and AFTER training with
the exact deployed inference (4 s windows over the whole clip, median).

Smoke test first:  python train_itw_finetune.py --smoke
Full run:          python train_itw_finetune.py --check_clips clip1.mp3 clip2.mp3
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from modules.content.audio_utils import preprocess_audio, TARGET_SAMPLE_RATE
from modules.content.audio_analyzer import _window_starts, MAX_AUDIO_SAMPLES, BATCH_SIZE
from train_wav2vec2 import (ASVspoofDataset, augment_waveform, freeze_lower_layers,
                            get_device, format_eta, parse_protocol_file,
                            LABEL_FAKE, LABEL_REAL)

SR = TARGET_SAMPLE_RATE
MP3_BITRATES_K = [48, 64, 96, 128]


# ---------------------------------------------------------------- split
def make_speaker_split(rows, n_seeds=2000, targets=(0.70, 0.15, 0.15), min_spk=5):
    """Speaker-disjoint train/val/test split. Among n_seeds seeded shuffles, keep the one
    whose clip fractions and bona-fide share are closest to the targets. The criterion
    uses only the metadata (never model results). Returns (train, val, test, seed)."""
    n_clips, n_real = {}, {}
    for r in rows:
        n_clips[r['speaker']] = n_clips.get(r['speaker'], 0) + 1
        n_real[r['speaker']] = n_real.get(r['speaker'], 0) + (r['label'] == 'bona-fide')
    spks = sorted(n_clips)
    total = sum(n_clips.values())
    overall_real = sum(n_real.values()) / total
    best = None
    for seed in range(n_seeds):
        order = spks[:]
        random.Random(seed).shuffle(order)
        i, test, val = 0, [], []
        cum = 0
        while i < len(order) and cum < targets[2] * total:
            cum += n_clips[order[i]]; test.append(order[i]); i += 1
        cum = 0
        while i < len(order) and cum < targets[1] * total:
            cum += n_clips[order[i]]; val.append(order[i]); i += 1
        train = order[i:]
        if min(len(train), len(val), len(test)) < min_spk:
            continue
        score = 0.0
        for grp, tgt in ((train, targets[0]), (val, targets[1]), (test, targets[2])):
            c = sum(n_clips[s] for s in grp)
            rshare = sum(n_real[s] for s in grp) / c
            score += abs(c / total - tgt) + abs(rshare - overall_real)
        if best is None or score < best[0]:
            best = (score, seed, train, val, test)
    if best is None:
        raise RuntimeError('No valid speaker split found')
    return sorted(best[2]), sorted(best[3]), sorted(best[4]), best[1]


def cap_per_class(items, n, seed):
    rng = random.Random(seed)
    out = []
    for lab in (LABEL_REAL, LABEL_FAKE):
        grp = sorted(i for i in items if i[1] == lab)
        rng.shuffle(grp)
        out += grp[:n]
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- augmentation
def mp3_roundtrip_np(x, bitrate_k):
    """16 kHz float32 -> 44.1 kHz mp3 at bitrate_k -> back to 16 kHz. None on failure."""
    raw = np.asarray(x, dtype='<f4').tobytes()
    enc = subprocess.run(
        ['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', '16000', '-ac', '1', '-i', 'pipe:0',
         '-ar', '44100', '-b:a', f'{bitrate_k}k', '-f', 'mp3', 'pipe:1'],
        input=raw, capture_output=True, timeout=30)
    if enc.returncode != 0 or not enc.stdout:
        return None
    dec = subprocess.run(
        ['ffmpeg', '-v', 'error', '-f', 'mp3', '-i', 'pipe:0',
         '-f', 'f32le', '-ar', '16000', '-ac', '1', 'pipe:1'],
        input=enc.stdout, capture_output=True, timeout=30)
    if dec.returncode != 0 or not dec.stdout:
        return None
    y = np.frombuffer(dec.stdout, dtype='<f4').copy()
    n = len(x)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def mp3_augment(w):
    try:
        y = mp3_roundtrip_np(w.numpy(), random.choice(MP3_BITRATES_K))
    except Exception:
        y = None
    return torch.from_numpy(y) if y is not None else w


class ITWDataset(Dataset):
    """Items are (path, label). Train mode: random 4 s crop for longer clips, optional
    mp3 round trip, then the same augment_waveform used for the ASVspoof model."""

    def __init__(self, items, train, mp3_prob=0.0, augment=True):
        self.samples = items
        self.train = train
        self.mp3_prob = mp3_prob
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        w = preprocess_audio(path)
        if w.shape[0] > MAX_AUDIO_SAMPLES:
            start = random.randint(0, w.shape[0] - MAX_AUDIO_SAMPLES) if self.train else 0
            w = w[start:start + MAX_AUDIO_SAMPLES]
        elif w.shape[0] < MAX_AUDIO_SAMPLES:
            w = torch.nn.functional.pad(w, (0, MAX_AUDIO_SAMPLES - w.shape[0]))
        if self.train and self.augment:
            if random.random() < self.mp3_prob:
                w = mp3_augment(w)
            w = augment_waveform(w, random)
        return w, label


# ---------------------------------------------------------------- evaluation
def auc_eer(y, s):
    order = np.argsort(-s, kind='mergesort')
    ys = y[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    tpr, fpr = np.r_[0, tps / tps[-1]], np.r_[0, fps / fps[-1]]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return auc, float((fnr[i] + fpr[i]) / 2)


def window_probs(model, fe, device, waveform):
    """Fake probability of every 4 s window, exactly as the deployed analyzer cuts them."""
    starts, _ = _window_starts(waveform.shape[0])
    chunks = []
    for s in starts:
        c = waveform[s:s + MAX_AUDIO_SAMPLES]
        if c.shape[0] < MAX_AUDIO_SAMPLES:
            c = torch.nn.functional.pad(c, (0, MAX_AUDIO_SAMPLES - c.shape[0]))
        chunks.append(c.numpy())
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH_SIZE):
            inp = fe(chunks[i:i + BATCH_SIZE], sampling_rate=SR, return_tensors='pt', padding=True)
            logits = model(input_values=inp['input_values'].to(device)).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 0].cpu().tolist())   # index 0 = fake
    return np.array(probs)


def evaluate(model, fe, device, items):
    ys, ps, failures = [], [], 0
    for path, label in items:
        try:
            p = float(np.median(window_probs(model, fe, device, preprocess_audio(path))))
        except Exception:
            failures += 1
            continue
        ys.append(int(label == LABEL_FAKE))
        ps.append(p)
    y, s = np.array(ys), np.array(ps)
    real, fake = s[y == 0], s[y == 1]
    m = {'n_real': int(len(real)), 'n_fake': int(len(fake)), 'load_failures': failures}
    if len(real) == 0 or len(fake) == 0:
        return m
    m['real_recall'] = float((real < 0.5).mean())
    m['fake_recall'] = float((fake >= 0.5).mean())
    m['bal_acc'] = (m['real_recall'] + m['fake_recall']) / 2
    m['auc'], m['eer'] = auc_eer(y, s)
    return m


def fmt(m):
    if 'bal_acc' not in m:
        return f"insufficient data {m}"
    return (f"real->REAL {100 * m['real_recall']:.1f}% | spoof->FAKE {100 * m['fake_recall']:.1f}% | "
            f"bal.acc {100 * m['bal_acc']:.1f}% | AUC {m['auc']:.3f} | EER {100 * m['eer']:.1f}% | "
            f"n={m['n_real']}+{m['n_fake']} (load failures {m['load_failures']})")


def clip_report(model, fe, device, paths):
    out = {}
    for p in paths:
        try:
            wp = window_probs(model, fe, device, preprocess_audio(p))
            out[os.path.basename(p)] = {
                'median': float(np.median(wp)), 'min': float(wp.min()), 'max': float(wp.max()),
                'windows_pct': [round(100 * v, 1) for v in wp]}
        except Exception as e:
            out[os.path.basename(p)] = {'error': str(e)}
    return out


def run_all_evals(model, fe, device, sets, clips):
    res = {'itw_test': evaluate(model, fe, device, sets['itw_test']),
           'asv_dev': evaluate(model, fe, device, sets['asv_dev'])}
    if clips:
        res['clips'] = clip_report(model, fe, device, clips)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--itw_dir', default=os.path.expanduser('~/Desktop/Deepfake_Detection/datasets/itw/release_in_the_wild'))
    ap.add_argument('--asv_root', default=os.path.expanduser('~/.cache/kagglehub/datasets/anishsarkar22/asvpoof-2019-dataset-la/versions/1'))
    ap.add_argument('--init_checkpoint', default=str(ROOT / 'checkpoints/wav2vec2_asvspoof_augmented_v2/epoch_2'))
    ap.add_argument('--output_dir', default=str(ROOT / 'checkpoints/wav2vec2_itw_ft_v1'))
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--unfreeze_top_layers', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--itw_train_per_class', type=int, default=4000)
    ap.add_argument('--asv_bonafide_fraction', type=float, default=1.0)
    ap.add_argument('--asv_spoof_fraction', type=float, default=0.15)
    ap.add_argument('--val_per_class', type=int, default=500)
    ap.add_argument('--test_per_class', type=int, default=1500)
    ap.add_argument('--asv_dev_per_class', type=int, default=300)
    ap.add_argument('--mp3_prob', type=float, default=0.3)
    ap.add_argument('--no_augment', action='store_true')
    ap.add_argument('--split_seeds', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--check_clips', nargs='*', default=[])
    ap.add_argument('--smoke', action='store_true', help='tiny end-to-end run to verify the pipeline')
    args = ap.parse_args()

    max_batches = None
    if args.smoke:
        args.itw_train_per_class, args.asv_bonafide_fraction, args.asv_spoof_fraction = 40, 0.02, 0.002
        args.val_per_class = args.test_per_class = args.asv_dev_per_class = 20
        args.epochs, max_batches = 1, 10
        args.output_dir = str(Path(args.output_dir).with_name(Path(args.output_dir).name + '_SMOKE'))
        print('[smoke] tiny run - numbers are meaningless, this only checks the pipeline')

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device()
    print(f'[setup] device={device}  init={args.init_checkpoint}\n[setup] output={args.output_dir}')
    if not os.path.isdir(args.init_checkpoint):
        sys.exit(f'Init checkpoint not found: {args.init_checkpoint}')
    if os.path.exists(os.path.join(args.output_dir, 'epoch_1')):
        sys.exit(f'{args.output_dir} already has checkpoints - choose a new --output_dir (nothing is overwritten)')

    # ---- ITW split
    meta = os.path.join(args.itw_dir, 'meta.csv')
    rows = list(csv.DictReader(open(meta)))
    train_spk, val_spk, test_spk, split_seed = make_speaker_split(rows, args.split_seeds)
    assert not (set(train_spk) & set(val_spk)) and not (set(train_spk) & set(test_spk)) and not (set(val_spk) & set(test_spk))
    spk2split = {s: 'train' for s in train_spk}
    spk2split.update({s: 'val' for s in val_spk}); spk2split.update({s: 'test' for s in test_spk})
    items = {'train': [], 'val': [], 'test': []}
    for r in rows:
        p = os.path.join(args.itw_dir, r['file'])
        if os.path.exists(p):
            items[spk2split[r['speaker']]].append((p, LABEL_REAL if r['label'] == 'bona-fide' else LABEL_FAKE))
    print(f'\n[split] speaker-disjoint, seed {split_seed} (chosen from {args.split_seeds} by clip-fraction/class-balance only)')
    for k, spk in (('train', train_spk), ('val', val_spk), ('test', test_spk)):
        nr = sum(1 for _, l in items[k] if l == LABEL_REAL)
        print(f'   {k:5s} speakers={len(spk):2d}  clips={len(items[k]):6d}  bona-fide={nr}  spoof={len(items[k]) - nr}')
    print('   val speakers :', ', '.join(val_spk))
    print('   test speakers:', ', '.join(test_spk))
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump({'seed': split_seed, 'train': train_spk, 'val': val_spk, 'test': test_spk},
              open(os.path.join(args.output_dir, 'split.json'), 'w'), indent=1)

    itw_train = cap_per_class(items['train'], args.itw_train_per_class, args.seed)
    sets = {'itw_val': cap_per_class(items['val'], args.val_per_class, args.seed),
            'itw_test': cap_per_class(items['test'], args.test_per_class, args.seed)}

    # ---- ASVspoof (train mix + dev regression sample)
    la = os.path.join(args.asv_root, 'LA')
    proto_dir = os.path.join(la, 'ASVspoof2019_LA_cm_protocols')
    asv_train = ASVspoofDataset(os.path.join(la, 'ASVspoof2019_LA_train', 'flac'),
                                os.path.join(proto_dir, 'ASVspoof2019.LA.cm.train.trn.txt'),
                                bonafide_fraction=args.asv_bonafide_fraction,
                                spoof_fraction=args.asv_spoof_fraction,
                                augment=not args.no_augment, seed=args.seed)
    dev_labels = parse_protocol_file(os.path.join(proto_dir, 'ASVspoof2019.LA.cm.dev.trl.txt'))
    dev_flac = os.path.join(la, 'ASVspoof2019_LA_dev', 'flac')
    dev_items = [(os.path.join(dev_flac, f'{n}.flac'), l) for n, l in dev_labels.items()
                 if os.path.exists(os.path.join(dev_flac, f'{n}.flac'))]
    sets['asv_dev'] = cap_per_class(dev_items, args.asv_dev_per_class, args.seed)

    itw_ds = ITWDataset(itw_train, train=True, mp3_prob=args.mp3_prob, augment=not args.no_augment)
    train_ds = ConcatDataset([asv_train, itw_ds])
    labels = [l for _, l in asv_train.samples] + [l for _, l in itw_ds.samples]
    n_real, n_fake = labels.count(LABEL_REAL), labels.count(LABEL_FAKE)
    print(f'\n[train set] ASVspoof {len(asv_train)} + ITW {len(itw_ds)} = {len(train_ds)} clips '
          f'({n_real} bona-fide, {n_fake} spoof) | mp3_prob={args.mp3_prob} | augment={not args.no_augment}')
    class_weights = torch.tensor([len(labels) / (2 * n_fake), len(labels) / (2 * n_real)], dtype=torch.float32).to(device)

    # ---- model + BEFORE
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.init_checkpoint)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.init_checkpoint).to(device)
    print('\n[BEFORE] scoring the starting checkpoint with deployed inference ...')
    before = run_all_evals(model, fe, device, sets, args.check_clips)
    print('   ITW test :', fmt(before['itw_test']))
    print('   ASV dev  :', fmt(before['asv_dev']))

    # ---- train
    freeze_lower_layers(model, args.unfreeze_top_layers)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    print(f'[train] {total_batches} batches/epoch, lr={args.lr}, top {args.unfreeze_top_layers} layers unfrozen')

    best = (-1.0, -1.0, None)   # (val bal_acc, val auc, epoch)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b, (wav, lab) in enumerate(loader, 1):
            if max_batches and b > max_batches:
                break
            inp = fe(wav.numpy(), sampling_rate=SR, return_tensors='pt', padding=True)['input_values'].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(input_values=inp).logits, lab.to(device))
            loss.backward()
            optimizer.step()
            tot += loss.item(); nb += 1
            if b % args.log_every == 0:
                el = time.time() - t0
                print(f'[epoch {epoch}] batch {b}/{total_batches} loss={loss.item():.4f} '
                      f'elapsed={format_eta(el)} ETA={format_eta(el / b * (total_batches - b))}', flush=True)
        print(f'[epoch {epoch}] train_loss={tot / max(nb, 1):.4f} time={format_eta(time.time() - t0)}')
        val = evaluate(model, fe, device, sets['itw_val'])
        print(f'[epoch {epoch}] ITW val : {fmt(val)}')
        ck = os.path.join(args.output_dir, f'epoch_{epoch}')
        model.save_pretrained(ck); fe.save_pretrained(ck)
        if 'bal_acc' in val and (val['bal_acc'], val['auc']) > (best[0], best[1]):
            best = (val['bal_acc'], val['auc'], epoch)
    if best[2] is None:
        sys.exit('No usable validation result - cannot select an epoch')
    print(f'\n[select] best epoch by ITW-val balanced accuracy: {best[2]} ({100 * best[0]:.1f}%)')

    # ---- AFTER (reload from disk so the saved checkpoint itself is what gets tested)
    best_dir = os.path.join(args.output_dir, f'epoch_{best[2]}')
    model = Wav2Vec2ForSequenceClassification.from_pretrained(best_dir).to(device)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(best_dir)
    after = run_all_evals(model, fe, device, sets, args.check_clips)

    print('\n=========== BEFORE vs AFTER (same held-out data, deployed inference) ===========')
    for k, name in (('itw_test', 'ITW test speakers (unseen)'), ('asv_dev', 'ASVspoof dev sample (regression)')):
        print(f'{name}\n   before: {fmt(before[k])}\n   after : {fmt(after[k])}')
    for c in args.check_clips:
        b, a = before['clips'].get(os.path.basename(c), {}), after['clips'].get(os.path.basename(c), {})
        print(os.path.basename(c))
        for tag, d in (('before', b), ('after ', a)):
            if 'error' in d:
                print(f'   {tag}: ERROR {d["error"]}')
            else:
                print(f'   {tag}: median fake prob {d["median"]:.3f} (min {d["min"]:.3f}, max {d["max"]:.3f}) windows%: {d["windows_pct"]}')
    json.dump({'best_epoch': best[2], 'before': before, 'after': after, 'args': vars(args)},
              open(os.path.join(args.output_dir, 'results.json'), 'w'), indent=1, default=str)
    print(f'\nSaved: {args.output_dir}/results.json   Best checkpoint: {best_dir}')
    print('To try it in the app: AUDIO_MODEL_CHECKPOINT=' + best_dir + ' python run.py')


if __name__ == '__main__':
    main()
