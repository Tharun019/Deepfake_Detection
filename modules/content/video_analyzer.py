import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor

# ── Config — must match training exactly (kaggle_video_train.py, face-crop
#    version) ──────────────────────────────────────────────────────────────
NUM_FRAMES = 16
LOCAL_BACKBONE_DIR = os.path.join(os.path.dirname(__file__), '../../models/video/videomae-base')
HUB_BACKBONE_ID = "MCG-NJU/videomae-base"
HEAD_CHECKPOINT = os.path.join(os.path.dirname(__file__), '../../models/video/video_head_facecrop.pth')
NORM_STATS_CHECKPOINT = os.path.join(os.path.dirname(__file__), '../../models/video/feature_norm_stats_facecrop.npz')

# Face-crop params — must match crop_to_face_np() in the training notebook
# (cell-3) exactly, which itself mirrors the image pipeline's face-crop logic.
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (60, 60)
FACE_MARGIN = 0.2

_backbone = None
_processor = None
_head = None
_face_cascade = None
_mu = None
_sigma = None


class _Head(nn.Module):
    """Must match the Head class in kaggle_video_train.py exactly, or
    load_state_dict will fail (or silently mismatch shapes)."""
    def __init__(self, in_dim=768, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _load_backbone():
    global _backbone, _processor
    if _backbone is not None:
        return
    source = LOCAL_BACKBONE_DIR if os.path.isdir(LOCAL_BACKBONE_DIR) else HUB_BACKBONE_ID
    _processor = VideoMAEImageProcessor.from_pretrained(source)
    _backbone = VideoMAEModel.from_pretrained(source)
    _backbone.eval()
    for p in _backbone.parameters():
        p.requires_grad = False


def _load_head():
    global _head, _mu, _sigma
    if _head is not None:
        return
    if not os.path.exists(HEAD_CHECKPOINT):
        raise FileNotFoundError(
            f"video_head_facecrop.pth not found at {HEAD_CHECKPOINT} — copy it "
            "(and feature_norm_stats_facecrop.npz) from the Kaggle notebook's "
            "Output tab into models/video/ before using this analyzer."
        )
    if not os.path.exists(NORM_STATS_CHECKPOINT):
        raise FileNotFoundError(
            f"feature_norm_stats_facecrop.npz not found at {NORM_STATS_CHECKPOINT} "
            "— required for standardizing embeddings the same way training did."
        )
    head = _Head()
    state = torch.load(HEAD_CHECKPOINT, map_location='cpu')
    head.load_state_dict(state)
    head.eval()
    _head = head

    stats = np.load(NORM_STATS_CHECKPOINT)
    _mu = stats['mu'].astype(np.float32)       # shape (1, 768)
    _sigma = stats['sigma'].astype(np.float32)  # shape (1, 768)


def _load_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return
    _face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )


def _crop_to_face(frame_rgb):
    """Detect the largest face in an RGB frame and crop to it with a 20%
    margin. Falls back to the full frame if no face is detected — same
    behavior as crop_to_face_np() in training, so train/inference stay
    consistent."""
    _load_face_cascade()
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    if len(faces) == 0:
        return frame_rgb

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    margin_x = int(w * FACE_MARGIN)
    margin_y = int(h * FACE_MARGIN)
    x0 = max(x - margin_x, 0)
    y0 = max(y - margin_y, 0)
    x1 = min(x + w + margin_x, frame_rgb.shape[1])
    y1 = min(y + h + margin_y, frame_rgb.shape[0])
    return frame_rgb[y0:y1, x0:x1]


def _extract_frames(video_path, num_frames=NUM_FRAMES):
    """Uniform frame sampling via OpenCV, with face-cropping applied per
    frame — must match extract_frames() in the training notebook (cell-3)
    exactly (same sampling scheme + same crop logic) to avoid a
    train/inference mismatch. Uses sequential grab()/retrieve() decoding —
    much faster than random-access seeking for compressed video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = set(np.linspace(0, max(total - 1, 0), num_frames, dtype=int).tolist())
    frames = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx in indices:
            ok, frame = cap.retrieve()
            if ok:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(_crop_to_face(frame_rgb))
        idx += 1
        if len(frames) >= num_frames:
            break
    cap.release()
    if len(frames) < num_frames:
        if not frames:
            return None
        while len(frames) < num_frames:
            frames.append(frames[-1])
    return frames[:num_frames]


def analyze_video(file_path):
    try:
        _load_backbone()
        _load_head()

        frames = _extract_frames(file_path)
        if frames is None:
            return {"score": 0.5, "error": "could not extract frames from video"}

        inputs = _processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)

        with torch.no_grad():
            out = _backbone(pixel_values=pixel_values)
            embedding = out.last_hidden_state.mean(dim=1)  # [1, 768]

            # Standardize using train-set mu/sigma — required, training used
            # standardized features and the head was never trained on raw ones.
            embedding_np = embedding.numpy()
            embedding_std = (embedding_np - _mu) / _sigma
            embedding_std = torch.from_numpy(embedding_std.astype(np.float32))

            logits = _head(embedding_std)
            probs = torch.softmax(logits, dim=1).squeeze()

        # label convention (fixed at training time): 0 = FAKE, 1 = REAL
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        fake_score = round(min(max(fake_prob, 0.05), 0.95), 4)

        return {
            "score": fake_score,
            "features": {
                "raw_fake_prob": round(fake_prob, 4),
                "raw_real_prob": round(real_prob, 4),
                "frames_used": NUM_FRAMES,
                "model": "VideoMAE-base (frozen, face-cropped) + trained linear-probe head",
            }
        }

    except Exception as e:
        return {"score": 0.5, "error": str(e)}
