"""
Fine-tune the ITW model (epoch_3) on MLAAD-tiny English, with ITW + ASVspoof replay.
Never overwrites existing checkpoints.

Held out from training:
  - Edge-TTS (en + de): same service as the external edge-tts test fakes
  - 8 English MLAAD TTS systems (seeded)  -> validation on unseen systems
  - ~15% of English real audio, book-disjoint -> validation
  - ITW val/test speakers: same split.json as wav2vec2_itw_ft_v1
  - German MLAAD -> report-only test

Smoke test: python train_mlaad_finetune.py --smoke
Full run:   python train_mlaad_finetune.py --fake_clips <8 files> --real_clips <6 files>
"""
import argparse, csv, json, os, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from modules.content.audio_utils import TARGET_SAMPLE_RATE
from train_wav2vec2 import (ASVspoofDataset, freeze_lower_layers, get_device, format_eta,
                            parse_protocol_file, LABEL_FAKE, LABEL_REAL)
from train_itw_finetune import ITWDataset, cap_per_class, evaluate, fmt, clip_report

SR = TARGET_SAMPLE_RATE
EXCLUDED_SYSTEMS = {'Edge-TTS'}


def book_of(path):
    name = Path(path).stem
    m = re.match(r'^(.*?)_\d+', name)
    return m.group(1) if m else name


def wavs(d):
    return sorted(str(p) for p in Path(d).glob('*.wav'))


def mlaad_splits(root, n_val_systems, val_real_frac, seed):
    fake_en = Path(root) / 'fake' / 'en'
    systems = sorted(d.name for d in fake_en.iterdir() if d.is_dir() and d.name not in EXCLUDED_SYSTEMS)
    val_sys = sorted(random.Random(seed).sample(systems, n_val_systems))
    train_sys = [s for s in systems if s not in val_sys]
    real = wavs(Path(root) / 'original' / 'en')
    per_book = {}
    for p in real:
        per_book[book_of(p)] = per_book.get(book_of(p), 0) + 1
    order = sorted(per_book)
    random.Random(seed).shuffle(order)
    val_books, cum = set(), 0
    for b in order:
        if cum >= val_real_frac * len(real):
            break
        val_books.add(b); cum += per_book[b]
    tr = [(p, LABEL_FAKE) for s in train_sys for p in wavs(fake_en / s)]
    tr += [(p, LABEL_REAL) for p in real if book_of(p) not in val_books]
    va = [(p, LABEL_FAKE) for s in val_sys for p in wavs(fake_en / s)]
    va += [(p, LABEL_REAL) for p in real if book_of(p) in val_books]
    fake_de = Path(root) / 'fake' / 'de'
    de = [(p, LABEL_FAKE) for d in sorted(fake_de.iterdir())
          if d.is_dir() and d.name not in EXCLUDED_SYSTEMS for p in wavs(d)]
    de += [(p, LABEL_REAL) for p in wavs(Path(root) / 'original' / 'de')]
    info = {'train_systems': train_sys, 'val_systems': val_sys,
            'val_books': sorted(val_books), 'excluded': sorted(EXCLUDED_SYSTEMS)}
    return tr, va, de, info


def itw_items(itw_dir, split_json):
    sp = json.load(open(split_json))
    spk2 = {s: k for k in ('train', 'val', 'test') for s in sp[k]}
    items = {'train': [], 'val': [], 'test': []}
    for r in csv.DictReader(open(os.path.join(itw_dir, 'meta.csv'))):
        p = os.path.join(itw_dir, r['file'])
        if os.path.exists(p) and r['speaker'] in spk2:
            items[spk2[r['speaker']]].append((p, LABEL_REAL if r['label'] == 'bona-fide' else LABEL_FAKE))
    return items


def count(items):
    nr = sum(1 for _, l in items if l == LABEL_REAL)
    return f'{len(items)} clips ({nr} real, {len(items) - nr} fake)'


def run_evals(model, fe, device, sets, clips):
    res = {k: evaluate(model, fe, device, sets[k]) for k in ('itw_test', 'asv_dev', 'mlaad_val', 'mlaad_de')}
    if clips:
        res['clips'] = clip_report(model, fe, device, clips)
    return res


def stop_rule(res, fake_clips, real_clips):
    c = res.get('clips', {})
    caught = sum(1 for p in fake_clips if c.get(os.path.basename(p), {}).get('median', 0.0) >= 0.5)
    flagged = sum(1 for p in real_clips if c.get(os.path.basename(p), {}).get('median', 1.0) >= 0.5)
    itw = res['itw_test'].get('bal_acc', 0.0)
    asv = res['asv_dev'].get('bal_acc', 0.0)
    return [(f'external fakes caught {caught}/{len(fake_clips)} (need >= 4)', caught >= 4),
            (f'real clips flagged {flagged}/{len(real_clips)} (need 0)', flagged == 0),
            (f'ITW test bal.acc {100 * itw:.1f}% (need >= 90.0)', itw >= 0.90),
            (f'ASVspoof dev bal.acc {100 * asv:.1f}% (need >= 95.0)', asv >= 0.95)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlaad_dir', default=os.path.expanduser('~/Desktop/datasets/MLAAD-tiny'))
    ap.add_argument('--itw_dir', default=os.path.expanduser('~/Desktop/Deepfake_Detection/datasets/itw/release_in_the_wild'))
    ap.add_argument('--asv_root', default=os.path.expanduser('~/.cache/kagglehub/datasets/anishsarkar22/asvpoof-2019-dataset-la/versions/1'))
    ap.add_argument('--init_checkpoint', default=str(ROOT / 'checkpoints/wav2vec2_itw_ft_v1/epoch_3'))
    ap.add_argument('--itw_split', default=str(ROOT / 'checkpoints/wav2vec2_itw_ft_v1/split.json'))
    ap.add_argument('--output_dir', default=str(ROOT / 'checkpoints/wav2vec2_mlaad_ft_v1'))
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--unfreeze_top_layers', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--n_val_systems', type=int, default=8)
    ap.add_argument('--val_real_frac', type=float, default=0.15)
    ap.add_argument('--itw_replay_per_class', type=int, default=1500)
    ap.add_argument('--asv_bonafide_fraction', type=float, default=0.5)
    ap.add_argument('--asv_spoof_fraction', type=float, default=0.05)
    ap.add_argument('--itw_val_per_class', type=int, default=500)
    ap.add_argument('--test_per_class', type=int, default=1500)
    ap.add_argument('--asv_dev_per_class', type=int, default=300)
    ap.add_argument('--mlaad_val_per_class', type=int, default=800)
    ap.add_argument('--de_per_class', type=int, default=1500)
    ap.add_argument('--mp3_prob', type=float, default=0.3)
    ap.add_argument('--no_augment', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--fake_clips', nargs='*', default=[])
    ap.add_argument('--real_clips', nargs='*', default=[])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    max_batches, mlaad_cap = None, None
    if args.smoke:
        args.itw_replay_per_class, args.asv_bonafide_fraction, args.asv_spoof_fraction = 20, 0.02, 0.002
        args.itw_val_per_class = args.test_per_class = args.asv_dev_per_class = 20
        args.mlaad_val_per_class = args.de_per_class = mlaad_cap = 20
        args.epochs, max_batches = 1, 10
        args.output_dir += '_SMOKE'
        print('[smoke] tiny run - numbers are meaningless, this only checks the pipeline')
    elif len(args.fake_clips) != 8 or len(args.real_clips) != 6:
        sys.exit('Full run needs the pre-declared external set: --fake_clips (8 files) --real_clips (6 files)')
    clips = args.fake_clips + args.real_clips
    for p in clips:
        if not os.path.exists(p):
            sys.exit(f'Missing clip: {p}')

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device()
    for need in (args.init_checkpoint, args.itw_split, args.mlaad_dir):
        if not os.path.exists(need):
            sys.exit(f'Not found: {need}')
    if os.path.exists(os.path.join(args.output_dir, 'epoch_1')):
        sys.exit(f'{args.output_dir} already has checkpoints - choose a new --output_dir')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[setup] device={device}\n[setup] init={args.init_checkpoint}\n[setup] output={args.output_dir}')

    # ---- MLAAD
    m_tr, m_va, m_de, info = mlaad_splits(args.mlaad_dir, args.n_val_systems, args.val_real_frac, args.seed)
    assert not any('Edge-TTS' in p for p, _ in m_tr + m_va + m_de), 'Edge-TTS leaked into MLAAD sets'
    if mlaad_cap:
        m_tr = cap_per_class(m_tr, mlaad_cap, args.seed)
    json.dump(info, open(os.path.join(args.output_dir, 'mlaad_split.json'), 'w'), indent=1)
    print(f'\n[mlaad] excluded: {info["excluded"]}')
    print(f'[mlaad] train: {len(info["train_systems"])} systems | {count(m_tr)}')
    print(f'[mlaad] val  : {len(info["val_systems"])} held-out systems {info["val_systems"]}')
    print(f'[mlaad]        {len(info["val_books"])} held-out books | {count(m_va)}')
    print(f'[mlaad] German test: {count(m_de)}')

    # ---- ITW (same speaker split as the epoch_3 run)
    itw = itw_items(args.itw_dir, args.itw_split)
    replay = cap_per_class(itw['train'], args.itw_replay_per_class, args.seed)
    sets = {'itw_val': cap_per_class(itw['val'], args.itw_val_per_class, args.seed),
            'itw_test': cap_per_class(itw['test'], args.test_per_class, args.seed),
            'mlaad_val': cap_per_class(m_va, args.mlaad_val_per_class, args.seed),
            'mlaad_de': cap_per_class(m_de, args.de_per_class, args.seed)}

    # ---- ASVspoof
    la = os.path.join(args.asv_root, 'LA')
    proto = os.path.join(la, 'ASVspoof2019_LA_cm_protocols')
    asv_train = ASVspoofDataset(os.path.join(la, 'ASVspoof2019_LA_train', 'flac'),
                                os.path.join(proto, 'ASVspoof2019.LA.cm.train.trn.txt'),
                                bonafide_fraction=args.asv_bonafide_fraction,
                                spoof_fraction=args.asv_spoof_fraction,
                                augment=not args.no_augment, seed=args.seed)
    dev_flac = os.path.join(la, 'ASVspoof2019_LA_dev', 'flac')
    dev = [(os.path.join(dev_flac, f'{n}.flac'), l)
           for n, l in parse_protocol_file(os.path.join(proto, 'ASVspoof2019.LA.cm.dev.trl.txt')).items()
           if os.path.exists(os.path.join(dev_flac, f'{n}.flac'))]
    sets['asv_dev'] = cap_per_class(dev, args.asv_dev_per_class, args.seed)

    aug = not args.no_augment
    parts = [ITWDataset(m_tr, train=True, mp3_prob=args.mp3_prob, augment=aug),
             ITWDataset(replay, train=True, mp3_prob=args.mp3_prob, augment=aug),
             asv_train]
    train_ds = ConcatDataset(parts)
    labels = [l for d in parts for _, l in d.samples]
    n_real, n_fake = labels.count(LABEL_REAL), labels.count(LABEL_FAKE)
    print(f'\n[train set] MLAAD {len(parts[0])} + ITW replay {len(parts[1])} + ASVspoof {len(parts[2])} '
          f'= {len(train_ds)} ({n_real} real, {n_fake} fake)')
    w = torch.tensor([len(labels) / (2 * n_fake), len(labels) / (2 * n_real)], dtype=torch.float32).to(device)

    # ---- BEFORE
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.init_checkpoint)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.init_checkpoint).to(device)
    print('\n[BEFORE] scoring epoch_3 with deployed inference ...')
    before = run_evals(model, fe, device, sets, clips)
    for k in ('itw_test', 'asv_dev', 'mlaad_val', 'mlaad_de'):
        print(f'   {k:9s}: {fmt(before[k])}')

    # ---- TRAIN
    freeze_lower_layers(model, args.unfreeze_top_layers)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    total = min(len(loader), max_batches) if max_batches else len(loader)
    print(f'[train] {total} batches/epoch, lr={args.lr}, top {args.unfreeze_top_layers} layers unfrozen')
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b, (wav, lab) in enumerate(loader, 1):
            if max_batches and b > max_batches:
                break
            inp = fe(wav.numpy(), sampling_rate=SR, return_tensors='pt', padding=True)['input_values'].to(device)
            opt.zero_grad()
            loss = loss_fn(model(input_values=inp).logits, lab.to(device))
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if b % args.log_every == 0:
                el = time.time() - t0
                print(f'[epoch {epoch}] batch {b}/{total} loss={loss.item():.4f} '
                      f'elapsed={format_eta(el)} ETA={format_eta(el / b * (total - b))}', flush=True)
        print(f'[epoch {epoch}] train_loss={tot / max(nb, 1):.4f} time={format_eta(time.time() - t0)}')
        mv = evaluate(model, fe, device, sets['mlaad_val'])
        iv = evaluate(model, fe, device, sets['itw_val'])
        print(f'[epoch {epoch}] MLAAD val: {fmt(mv)}\n[epoch {epoch}] ITW val  : {fmt(iv)}')
        ck = os.path.join(args.output_dir, f'epoch_{epoch}')
        model.save_pretrained(ck); fe.save_pretrained(ck)
        if 'bal_acc' in mv and 'bal_acc' in iv:
            key = ((mv['bal_acc'] + iv['bal_acc']) / 2, (mv['auc'] + iv['auc']) / 2)
            if best is None or key > best[0]:
                best = (key, epoch)
    if best is None:
        sys.exit('No usable validation result - cannot select an epoch')
    print(f'\n[select] best epoch by mean(MLAAD-val, ITW-val) bal.acc: {best[1]} ({100 * best[0][0]:.1f}%)')

    # ---- AFTER
    best_dir = os.path.join(args.output_dir, f'epoch_{best[1]}')
    model = Wav2Vec2ForSequenceClassification.from_pretrained(best_dir).to(device)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(best_dir)
    after = run_evals(model, fe, device, sets, clips)

    print('\n=========== BEFORE (epoch_3) vs AFTER (selected epoch) ===========')
    for k, name in (('itw_test', 'ITW test speakers'), ('asv_dev', 'ASVspoof dev'),
                    ('mlaad_val', 'MLAAD val (unseen systems)'), ('mlaad_de', 'MLAAD German (report only)')):
        print(f'{name}\n   before: {fmt(before[k])}\n   after : {fmt(after[k])}')

    def s(d):
        return 'ERROR' if not d or 'error' in d else f"{100 * d['median']:.1f}% (max window {100 * d['max']:.1f}%)"
    for p in clips:
        n = os.path.basename(p)
        tag = 'FAKE' if p in args.fake_clips else 'REAL'
        print(f'[{tag}] {n}\n   before: {s(before.get("clips", {}).get(n))}\n   after : {s(after.get("clips", {}).get(n))}')

    if not args.smoke:
        print('\n=========== STOP RULE (pre-declared) ===========')
        checks = stop_rule(after, args.fake_clips, args.real_clips)
        for txt, ok in checks:
            print(f'   {"PASS" if ok else "FAIL"}  {txt}')
        print('   VERDICT:', 'KEEP new model' if all(ok for _, ok in checks) else 'REJECT - keep epoch_3, document limitation')

    json.dump({'best_epoch': best[1], 'before': before, 'after': after, 'mlaad_split': info, 'args': vars(args)},
              open(os.path.join(args.output_dir, 'results.json'), 'w'), indent=1, default=str)
    print(f'\nSaved: {args.output_dir}/results.json   Best checkpoint: {best_dir}')


if __name__ == '__main__':
    main()
