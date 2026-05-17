import subprocess
import tempfile
import os
import torch
import numpy as np
from transformers import VideoMAEModel, AutoImageProcessor
from PIL import Image

MODEL_PATH = "models/video/videomae-base"

_model = None
_processor = None

def _load_model():
    global _model, _processor
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        _model = VideoMAEModel.from_pretrained(MODEL_PATH)
        _model.eval()

def extract_frames_ffmpeg(video_path, num_frames=16):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select=not(mod(n\\,10))",
            "-vsync", "vfr",
            "-q:v", "2",
            "-frames:v", str(num_frames * 3),
            output_pattern,
            "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)
        frame_files = sorted([
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if f.endswith(".jpg")
        ])
        if len(frame_files) < num_frames:
            return None
        indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            img = Image.open(frame_files[idx]).convert("RGB")
            frames.append(np.array(img))
        return frames

def analyze_video(file_path):
    try:
        _load_model()

        frames = extract_frames_ffmpeg(file_path, num_frames=16)
        if frames is None:
            return {"score": 0.5, "error": "could not extract frames"}

        inputs = _processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)

        with torch.no_grad():
            outputs = _model(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state

        features = hidden[0].numpy()

        mean_norm = float(np.linalg.norm(features, axis=1).mean())
        variance = float(np.var(features))
        sparsity = float(np.mean(np.abs(features) < 0.01))
        temporal_std = float(np.std(features, axis=0).mean())

        norm_score = min(mean_norm / 80.0, 1.0)
        var_score = min(variance / 0.5, 1.0)
        sparsity_score = sparsity
        temporal_score = min(temporal_std / 1.5, 1.0)

        fake_score = (
            0.30 * norm_score +
            0.25 * var_score +
            0.20 * sparsity_score +
            0.25 * temporal_score
        )

        fake_score = float(np.clip(fake_score, 0.0, 1.0))

        return {
            "score": round(fake_score, 4),
            "features": {
                "mean_norm": round(mean_norm, 4),
                "variance": round(variance, 4),
                "sparsity": round(sparsity, 4),
                "temporal_std": round(temporal_std, 4)
            }
        }

    except Exception as e:
        return {"score": 0.5, "error": str(e)}
