"""
Fine-tuning script for Wav2Vec2ForSequenceClassification on ASVspoof 2019 LA.
Labeling: 0 = fake/spoof, 1 = real/bonafide.
Optional --augment: stronger real-world recording mismatch simulation
(90% resample round-trip, 80% noise injection, 70% gain jitter).
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.content.audio_utils import preprocess_audio, TARGET_SAMPLE_RATE

MAX_AUDIO_SECONDS = 4
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE
LABEL_FAKE = 0
LABEL_REAL = 1
_AUGMENT_RATES = [8000, 22050, 32000, 44100, 48000]


def parse_protocol_file(protocol_path: str) -> dict:
    labels = {}
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, filename, _, _, label_str = parts
            if label_str == "bonafide":
                labels[filename] = LABEL_REAL
            elif label_str == "spoof":
                labels[filename] = LABEL_FAKE
            else:
                raise ValueError(f"Unexpected label '{label_str}'")
    return labels


def augment_waveform(waveform: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if rng.random() < 0.9:
        intermediate_rate = rng.choice(_AUGMENT_RATES)
        if intermediate_rate != TARGET_SAMPLE_RATE:
            up = torchaudio.transforms.Resample(TARGET_SAMPLE_RATE, intermediate_rate)
            down = torchaudio.transforms.Resample(intermediate_rate, TARGET_SAMPLE_RATE)
            waveform = down(up(waveform.unsqueeze(0))).squeeze(0)
            if waveform.shape[0] > MAX_AUDIO_SAMPLES:
                waveform = waveform[:MAX_AUDIO_SAMPLES]
            elif waveform.shape[0] < MAX_AUDIO_SAMPLES:
                waveform = torch.nn.functional.pad(waveform, (0, MAX_AUDIO_SAMPLES - waveform.shape[0]))

    if rng.random() < 0.8:
        snr_db = rng.uniform(10, 30)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power + 1e-10)
        waveform = waveform + noise

    if rng.random() < 0.7:
        gain_db = rng.uniform(-8, 8)
        gain_factor = 10 ** (gain_db / 20)
        waveform = waveform * gain_factor

    waveform = torch.clamp(waveform, -1.0, 1.0)
    return waveform


class ASVspoofDataset(Dataset):
    def __init__(self, flac_dir, protocol_path, bonafide_fraction=1.0, spoof_fraction=1.0,
                 augment=False, seed=42):
        self.flac_dir = Path(flac_dir)
        self.labels_map = parse_protocol_file(protocol_path)
        self.augment = augment
        self.samples = []
        missing = 0
        for filename, label in self.labels_map.items():
            flac_path = self.flac_dir / f"{filename}.flac"
            if flac_path.exists():
                self.samples.append((flac_path, label))
            else:
                missing += 1
        if missing > 0:
            print(f"[ASVspoofDataset] Warning: {missing} files missing from {flac_dir}")
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for {flac_dir}")

        rng = random.Random(seed)
        real_samples = [s for s in self.samples if s[1] == LABEL_REAL]
        fake_samples = [s for s in self.samples if s[1] == LABEL_FAKE]
        rng.shuffle(real_samples)
        rng.shuffle(fake_samples)
        n_real = max(1, round(len(real_samples) * bonafide_fraction))
        n_fake = max(1, round(len(fake_samples) * spoof_fraction))
        subset = real_samples[:n_real] + fake_samples[:n_fake]
        rng.shuffle(subset)
        self.samples = subset
        self._aug_rng = random.Random(seed + 1)

        n_real_final = sum(1 for _, l in self.samples if l == LABEL_REAL)
        n_fake_final = sum(1 for _, l in self.samples if l == LABEL_FAKE)
        print(f"[ASVspoofDataset] {flac_dir}: {len(self.samples)} samples "
              f"({n_real_final} bonafide, {n_fake_final} spoof) | augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flac_path, label = self.samples[idx]
        waveform = preprocess_audio(str(flac_path))
        if waveform.shape[0] > MAX_AUDIO_SAMPLES:
            waveform = waveform[:MAX_AUDIO_SAMPLES]
        elif waveform.shape[0] < MAX_AUDIO_SAMPLES:
            pad_amount = MAX_AUDIO_SAMPLES - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        if self.augment:
            waveform = augment_waveform(waveform, self._aug_rng)
        return waveform, label


def compute_class_weights(dataset: ASVspoofDataset) -> torch.Tensor:
    n_real = sum(1 for _, l in dataset.samples if l == LABEL_REAL)
    n_fake = sum(1 for _, l in dataset.samples if l == LABEL_FAKE)
    total = n_real + n_fake
    weight_fake = total / (2 * n_fake) if n_fake > 0 else 1.0
    weight_real = total / (2 * n_real) if n_real > 0 else 1.0
    return torch.tensor([weight_fake, weight_real], dtype=torch.float32)


def freeze_lower_layers(model, unfreeze_top_n: int):
    model.freeze_feature_encoder()
    transformer_layers = model.wav2vec2.encoder.layers
    num_layers = len(transformer_layers)
    unfreeze_from = max(0, num_layers - unfreeze_top_n)
    for i, layer in enumerate(transformer_layers):
        requires_grad = i >= unfreeze_from
        for param in layer.parameters():
            param.requires_grad = requires_grad
    print(f"[freeze_lower_layers] fine-tuning top {unfreeze_top_n} of {num_layers} layers")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}hr"


def train(args):
    device = get_device()
    print(f"[train] Using device: {device}")

    train_flac_dir = os.path.join(args.data_root, "LA", "ASVspoof2019_LA_train", "flac")
    dev_flac_dir = os.path.join(args.data_root, "LA", "ASVspoof2019_LA_dev", "flac")
    train_protocol = os.path.join(args.data_root, "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
    dev_protocol = os.path.join(args.data_root, "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")

    train_dataset = ASVspoofDataset(train_flac_dir, train_protocol,
                                     bonafide_fraction=args.train_bonafide_fraction,
                                     spoof_fraction=args.train_spoof_fraction,
                                     augment=args.augment)
    dev_dataset = ASVspoofDataset(dev_flac_dir, dev_protocol,
                                   bonafide_fraction=args.dev_bonafide_fraction,
                                   spoof_fraction=args.dev_spoof_fraction,
                                   augment=False)

    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"[train] Class weights [fake, real]: {class_weights.tolist()}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    total_train_batches = len(train_loader)
    print(f"[train] {total_train_batches} batches per epoch (batch_size={args.batch_size})")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
    freeze_lower_layers(model, args.unfreeze_top_layers)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        last_log_time = epoch_start

        for batch_idx, (waveforms, labels) in enumerate(train_loader, start=1):
            inputs = feature_extractor(waveforms.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_values=input_values)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            now = time.time()
            if batch_idx % args.log_every == 0 or (now - last_log_time) > 10:
                elapsed = now - epoch_start
                bps = batch_idx / elapsed if elapsed > 0 else 0
                remaining = total_train_batches - batch_idx
                eta_sec = remaining / bps if bps > 0 else 0
                print(f"[epoch {epoch}] batch {batch_idx}/{total_train_batches} | loss={loss.item():.4f} "
                      f"| {bps:.2f} batches/sec | elapsed={format_eta(elapsed)} | ETA={format_eta(eta_sec)}", flush=True)
                last_log_time = now

        avg_train_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start
        print(f"[epoch {epoch}] train_loss={avg_train_loss:.4f} | epoch_time={format_eta(epoch_time)}")

        model.eval()
        real_correct = real_total = 0
        fake_correct = fake_total = 0
        with torch.no_grad():
            for waveforms, labels in dev_loader:
                inputs = feature_extractor(waveforms.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
                input_values = inputs["input_values"].to(device)
                labels = labels.to(device)
                outputs = model(input_values=input_values)
                preds = torch.argmax(outputs.logits, dim=-1)

                real_mask = labels == LABEL_REAL
                fake_mask = labels == LABEL_FAKE
                real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
                real_total += real_mask.sum().item()
                fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
                fake_total += fake_mask.sum().item()

        bonafide_recall = real_correct / max(real_total, 1)
        spoof_recall = fake_correct / max(fake_total, 1)
        overall_acc = (real_correct + fake_correct) / max(real_total + fake_total, 1)
        print(f"[epoch {epoch}] bonafide_recall={bonafide_recall:.4f} ({real_correct}/{real_total}) "
              f"| spoof_recall={spoof_recall:.4f} ({fake_correct}/{fake_total}) "
              f"| overall_accuracy={overall_acc:.4f}")

        checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch}")
        model.save_pretrained(checkpoint_path)
        feature_extractor.save_pretrained(checkpoint_path)
        print(f"[epoch {epoch}] checkpoint saved to {checkpoint_path}")

    print("[train] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_top_layers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="checkpoints/wav2vec2_asvspoof")
    parser.add_argument("--train_bonafide_fraction", type=float, default=1.0)
    parser.add_argument("--train_spoof_fraction", type=float, default=1.0)
    parser.add_argument("--dev_bonafide_fraction", type=float, default=1.0)
    parser.add_argument("--dev_spoof_fraction", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    train(args)
