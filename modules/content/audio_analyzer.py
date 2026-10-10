"""
L2 audio content analyzer — Wav2Vec2-based spoof/bonafide classification.
Score convention: higher = more likely fake (matches _analyze_image).

Windowed inference: the model was trained on 4 s inputs, so the clip is cut
into 4 s windows, each window is scored independently, and the per-window
fake probabilities are aggregated into one file-level score.
"""

import os

import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

from modules.content.audio_utils import preprocess_audio, TARGET_SAMPLE_RATE

MAX_AUDIO_SECONDS = 4          # window length the model was trained on
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE
MIN_TAIL_SAMPLES = TARGET_SAMPLE_RATE   # leftover shorter than 1 s is ignored
MAX_WINDOWS = 30               # safety cap (~2 min); evenly subsampled beyond this
BATCH_SIZE = 8
# 'median' (robust to isolated outlier windows) or 'mean'. Choose once, a priori,
# and evaluate on the full set of known clips — not per-file.
AGGREGATION = os.environ.get('AUDIO_AGGREGATION', 'median')

_DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '../../checkpoints/wav2vec2_itw_ft_v1/epoch_3'
)
_CHECKPOINT_DIR = os.environ.get('AUDIO_MODEL_CHECKPOINT', _DEFAULT_CHECKPOINT_DIR)

_audio_model = None
_audio_feature_extractor = None
_audio_device = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _get_audio_model():
    global _audio_model, _audio_feature_extractor, _audio_device

    if _audio_model is not None:
        return _audio_model, _audio_feature_extractor, _audio_device

    if not os.path.exists(_CHECKPOINT_DIR):
        raise FileNotFoundError(
            f"Audio model checkpoint not found at {_CHECKPOINT_DIR}. "
            f"Run train_wav2vec2.py first, or set AUDIO_MODEL_CHECKPOINT."
        )

    _audio_device = _get_device()
    _audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_CHECKPOINT_DIR)
    _audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(_CHECKPOINT_DIR)
    _audio_model.to(_audio_device)
    _audio_model.eval()

    return _audio_model, _audio_feature_extractor, _audio_device


def _window_starts(n_samples: int, win: int = MAX_AUDIO_SAMPLES,
                   min_tail: int = MIN_TAIL_SAMPLES, max_windows: int = MAX_WINDOWS):
    """Start indices of consecutive non-overlapping windows covering the clip.
    If a leftover >= min_tail remains, one extra window aligned to the end of
    the clip is added (overlaps the previous window, avoids zero-padding).
    Returns (starts_used, total_before_cap)."""
    if n_samples <= win:
        return [0], 1
    starts = list(range(0, n_samples - win + 1, win))
    if n_samples - (starts[-1] + win) >= min_tail:
        starts.append(n_samples - win)
    total = len(starts)
    if total > max_windows:
        idx = np.linspace(0, total - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx.tolist()))]
    return starts, total


def analyze_audio(file_path: str) -> dict:
    try:
        model, feature_extractor, device = _get_audio_model()

        waveform = preprocess_audio(file_path)
        original_duration_sec = waveform.shape[0] / TARGET_SAMPLE_RATE

        starts, total_windows = _window_starts(waveform.shape[0])

        chunks = []
        for s in starts:
            chunk = waveform[s:s + MAX_AUDIO_SAMPLES]
            if chunk.shape[0] < MAX_AUDIO_SAMPLES:   # only for clips < 4 s
                chunk = torch.nn.functional.pad(chunk, (0, MAX_AUDIO_SAMPLES - chunk.shape[0]))
            chunks.append(chunk.numpy())

        window_fake_probs = []
        for i in range(0, len(chunks), BATCH_SIZE):
            inputs = feature_extractor(
                chunks[i:i + BATCH_SIZE], sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors='pt', padding=True,
            )
            input_values = inputs['input_values'].to(device)
            with torch.no_grad():
                logits = model(input_values=input_values).logits
                probs = torch.softmax(logits, dim=-1)
            window_fake_probs.extend(probs[:, 0].cpu().tolist())   # label 0 = fake

        arr = np.array(window_fake_probs, dtype=np.float64)
        fake_prob = float(np.mean(arr)) if AGGREGATION == 'mean' else float(np.median(arr))
        real_prob = 1.0 - fake_prob
        clamped_score = round(min(max(fake_prob, 0.05), 0.95), 4)

        return {
            'score': clamped_score,
            'features': {
                'raw_fake_prob': round(fake_prob, 4),
                'raw_real_prob': round(real_prob, 4),
                'aggregation': AGGREGATION,
                'windows_analyzed': len(starts),
                'window_fake_min': round(float(arr.min()), 4),
                'window_fake_max': round(float(arr.max()), 4),
                'window_fake_mean': round(float(arr.mean()), 4),
                'window_fake_probs_pct': ', '.join(f'{p * 100:.1f}' for p in arr),
                'analyzed_window_sec': MAX_AUDIO_SECONDS,
                'original_duration_sec': round(original_duration_sec, 2),
                'truncated': total_windows > len(starts),
            },
        }
    except Exception as e:
        return {'score': 0.5, 'features': {'error': str(e)}}
