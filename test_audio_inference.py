"""
Quick sanity-check inference script for the fine-tuned Wav2Vec2 audio model.
"""

import sys
from pathlib import Path

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.content.audio_utils import preprocess_audio, TARGET_SAMPLE_RATE

MAX_AUDIO_SECONDS = 4
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE

LABEL_NAMES = {0: "FAKE (spoof)", 1: "REAL (bonafide)"}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_audio_inference.py <audio_file> <checkpoint_dir>")
        sys.exit(1)

    audio_path = sys.argv[1]
    checkpoint_dir = sys.argv[2]

    device = get_device()
    print(f"[test_audio_inference] Using device: {device}")
    print(f"[test_audio_inference] Loading checkpoint from: {checkpoint_dir}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    print(f"[test_audio_inference] Loading audio: {audio_path}")
    waveform = preprocess_audio(audio_path)
    original_duration = waveform.shape[0] / TARGET_SAMPLE_RATE
    print(f"[test_audio_inference] Original duration: {original_duration:.2f} sec ({waveform.shape[0]} samples)")

    if waveform.shape[0] > MAX_AUDIO_SAMPLES:
        print(f"[test_audio_inference] WARNING: input longer than {MAX_AUDIO_SECONDS}s training window — "
              f"truncating to first {MAX_AUDIO_SECONDS}s for this check. Real inference on long audio "
              f"needs a separate chunking strategy (not yet built).")
        waveform = waveform[:MAX_AUDIO_SAMPLES]
    elif waveform.shape[0] < MAX_AUDIO_SAMPLES:
        pad_amount = MAX_AUDIO_SAMPLES - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    inputs = feature_extractor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values=input_values)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        pred_label = torch.argmax(probs).item()

    print("\n--- Result ---")
    print(f"File: {audio_path}")
    print(f"Window analyzed: first {MAX_AUDIO_SECONDS}s of {original_duration:.2f}s total")
    print(f"P(fake/spoof)     = {probs[0].item():.4f}")
    print(f"P(real/bonafide)  = {probs[1].item():.4f}")
    print(f"Predicted: {LABEL_NAMES[pred_label]}")


if __name__ == "__main__":
    main()
