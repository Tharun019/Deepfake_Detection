"""
L2 audio content analyzer — Wav2Vec2-based spoof/bonafide classification.
Score convention: higher = more likely fake (matches _analyze_image).
"""

import os

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

from modules.content.audio_utils import preprocess_audio, TARGET_SAMPLE_RATE

MAX_AUDIO_SECONDS = 4
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE

_DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '../../checkpoints/wav2vec2_asvspoof_augmented_v2/epoch_2'
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


def analyze_audio(file_path: str) -> dict:
    try:
        model, feature_extractor, device = _get_audio_model()

        waveform = preprocess_audio(file_path)
        original_duration_sec = waveform.shape[0] / TARGET_SAMPLE_RATE
        truncated = waveform.shape[0] > MAX_AUDIO_SAMPLES

        if truncated:
            waveform = waveform[:MAX_AUDIO_SAMPLES]
        elif waveform.shape[0] < MAX_AUDIO_SAMPLES:
            pad_amount = MAX_AUDIO_SAMPLES - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        inputs = feature_extractor(
            waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors='pt', padding=True,
        )
        input_values = inputs['input_values'].to(device)

        with torch.no_grad():
            outputs = model(input_values=input_values)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        clamped_score = round(min(max(fake_prob, 0.05), 0.95), 4)

        return {
            'score': clamped_score,
            'features': {
                'raw_fake_prob': round(fake_prob, 4),
                'raw_real_prob': round(real_prob, 4),
                'analyzed_window_sec': MAX_AUDIO_SECONDS,
                'original_duration_sec': round(original_duration_sec, 2),
                'truncated': truncated,
            },
        }
    except Exception as e:
        return {'score': 0.5, 'features': {'error': str(e)}}
