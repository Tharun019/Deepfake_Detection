import os
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

_audio_model = None
_audio_processor = None

def _get_audio_model():
    global _audio_model, _audio_processor
    if _audio_model is not None:
        return _audio_model, _audio_processor
    model_path = os.path.join(os.path.dirname(__file__), '../../models/audio/wav2vec2-base')
    _audio_processor = Wav2Vec2Processor.from_pretrained(model_path)
    _audio_model = Wav2Vec2Model.from_pretrained(model_path)
    _audio_model.eval()
    return _audio_model, _audio_processor

def _load_audio(file_path: str):
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-i', file_path,
            '-ar', '16000',
            '-ac', '1',
            '-f', 'f32le',
            '-hide_banner', '-loglevel', 'error',
            'pipe:1'
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or len(result.stdout) == 0:
            return None
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio
    except Exception:
        return None

def analyze_audio(file_path: str) -> dict:
    try:
        model, processor = _get_audio_model()
        audio = _load_audio(file_path)
        if audio is None or len(audio) < 1600:
            return {"score": 0.5}

        # Use max 5 seconds
        max_samples = 16000 * 5
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state.squeeze(0).numpy()

        scores = []

        # Heuristic 1: mean activation norm
        mean_norm = float(np.mean(np.linalg.norm(hidden, axis=1)))
        scores.append(min(mean_norm / 25.0, 1.0))

        # Heuristic 2: temporal variance (real speech varies more)
        temporal_var = float(np.mean(np.var(hidden, axis=0)))
        scores.append(1.0 - min(temporal_var / 2.0, 1.0))

        # Heuristic 3: feature sparsity
        sparsity = float(np.mean(hidden == 0))
        scores.append(1.0 - sparsity)

        # Heuristic 4: inter-frame cosine similarity
        # Synthetic audio tends to be more uniform across frames
        norms = np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-8
        normed = hidden / norms
        sim_matrix = normed @ normed.T
        upper = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
        mean_sim = float(np.mean(upper))
        scores.append(mean_sim)

        score = float(np.mean(scores))
        return {"score": round(min(max(score, 0.05), 0.95), 4)}

    except Exception:
        return {"score": 0.5}
