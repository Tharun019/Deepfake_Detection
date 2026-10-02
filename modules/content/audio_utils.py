"""
Shared audio preprocessing utility for DeepSentinel L2 audio module.

CRITICAL: This exact function must be imported by BOTH the training script
and the inference pipeline. Do not duplicate this logic anywhere else —
any divergence (different resampling method, different mono-mixing) causes
a train/inference mismatch that silently degrades real-world accuracy.

Wav2Vec2 requires 16kHz mono input because it was pretrained on LibriSpeech
(16kHz). Feeding it audio at any other sample rate without this conversion
will produce degraded or invalid results.

Uses soundfile for loading (avoids torchaudio's torchcodec backend
dependency) and torchaudio for resampling (pure torch op, no extra backend
needed).
"""

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000


def _convert_with_ffmpeg(filepath: str):
    """Fallback for containers soundfile/libsndfile can't decode directly
    (mp3, m4a/AAC — the default export format for macOS Voice Memos and
    QuickTime). Converts to a temp 16kHz mono WAV via ffmpeg.

    Returns the temp WAV path on success, or None if ffmpeg is missing or
    conversion fails (caller decides how to surface that).
    """
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", filepath,
             "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", tmp_wav],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0 or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
            return None
        return tmp_wav
    except Exception:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        return None


def preprocess_audio(filepath: str) -> torch.Tensor:
    """
    Load an audio file and convert it to 16kHz mono, ready for
    Wav2Vec2FeatureExtractor.

    Args:
        filepath: path to an audio file. soundfile loads wav/flac/ogg
            directly. mp3/m4a (the default export from Voice Memos/
            QuickTime on Mac) are not decodable by soundfile's libsndfile
            backend — those go through an automatic ffmpeg fallback
            below, converting to a temp 16kHz mono WAV first. This only
            triggers when the direct load fails, so wav/flac inputs
            (including all training data) are completely unaffected —
            no train/inference mismatch risk.

    Returns:
        1D torch.Tensor of shape (num_samples,), dtype float32,
        sampled at 16000 Hz, single channel.

    Raises:
        RuntimeError: if the file can't be loaded/decoded even via the
            ffmpeg fallback (e.g. ffmpeg isn't installed, or the file is
            genuinely corrupt).
    """
    converted_path = None
    try:
        data, original_sample_rate = sf.read(filepath, dtype="float32")
    except Exception as e:
        converted_path = _convert_with_ffmpeg(filepath)
        if converted_path is None:
            raise RuntimeError(
                f"Failed to load audio file '{filepath}': {e}. "
                f"ffmpeg fallback also failed — confirm ffmpeg is "
                f"installed (`ffmpeg -version`) and the file isn't corrupt."
            )
        try:
            data, original_sample_rate = sf.read(converted_path, dtype="float32")
        finally:
            os.remove(converted_path)

    # soundfile returns shape (num_samples,) for mono or
    # (num_samples, num_channels) for multi-channel.
    if data.ndim > 1:
        data = data.mean(axis=1)

    waveform = torch.from_numpy(data).unsqueeze(0)  # shape (1, num_samples)

    # Resample to 16kHz if not already at target rate.
    if original_sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
        waveform = resampler(waveform)

    # Drop the channel dimension -> (num_samples,), float32.
    waveform = waveform.squeeze(0).to(torch.float32)

    return waveform


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python audio_utils.py <path_to_audio_file>")
        sys.exit(1)

    test_path = sys.argv[1]
    wav = preprocess_audio(test_path)
    print(f"Loaded: {test_path}")
    print(f"Shape: {wav.shape}")
    print(f"Sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Duration: {wav.shape[0] / TARGET_SAMPLE_RATE:.2f} sec")
    print(f"Min/Max amplitude: {wav.min().item():.4f} / {wav.max().item():.4f}")
