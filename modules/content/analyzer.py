import io
import os
import base64
import torch
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import pillow_heif

pillow_heif.register_heif_opener()

_image_model = None
_face_cascade = None


def _get_image_model():
    global _image_model
    if _image_model is not None:
        return _image_model
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
    weights_path = os.path.join(os.path.dirname(__file__), '../../models/image/efficientnet_b4.pth')
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
    model.eval()
    _image_model = model
    return model


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _crop_to_face(img: Image.Image) -> Image.Image:
    """Detect largest face and crop with margin. Falls back to full frame if none found."""
    try:
        cascade = _get_face_cascade()
        img_np = np.array(img.convert('L'))  # grayscale for detection
        faces = cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            return img  # fallback — no face found, use full frame

        # pick largest face by area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # add 20% margin on each side
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        img_w, img_h = img.size

        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(img_w, x + w + margin_x)
        y1 = min(img_h, y + h + margin_y)

        return img.crop((x0, y0, x1, y1))
    except Exception:
        return img  # fallback on any detection error


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def analyze(file_path: str, media_type: str) -> float:
    if media_type == 'image':
        return _analyze_image(file_path)
    elif media_type == 'video':
        return _analyze_video(file_path)
    elif media_type == 'audio':
        return _analyze_audio(file_path)
    return 0.5


def _analyze_image(file_path: str) -> float:
    try:
        model = _get_image_model()
        img = Image.open(file_path).convert('RGB')
        img = _crop_to_face(img)
        tensor = _transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
        # outputs shape: [1, 2] — [fake_prob, real_prob]
        probs = torch.softmax(outputs, dim=1).squeeze()
        # class 0 fake, class 1 = real (ImageFolder sorts alphabetically)
        fake_prob = float(probs[0])
        return round(min(max(fake_prob, 0.05), 0.95), 4)
    except Exception:
        return 0.5


def _analyze_video(file_path: str) -> float:
    try:
        from modules.content.video_analyzer import analyze_video
        result = analyze_video(file_path)
        return result.get("score", 0.5)
    except Exception:
        return 0.5


def _analyze_audio(file_path: str) -> float:
    try:
        from modules.content.audio_analyzer import analyze_audio
        result = analyze_audio(file_path)
        return result.get("score", 0.5)
    except Exception:
        return 0.5


def generate_gradcam(file_path: str):
    """Grad-CAM heatmap for the fake class via EfficientNet-B4's conv_head.
    Returns a base64-encoded JPEG string, or None on failure.
    """
    try:
        model = _get_image_model()

        # ── Capture the feature map via a forward hook ──────────────────
        feature_map = [None]

        def _save_fmap(module, inp, out):
            feature_map[0] = out          # keep in computation graph

        target  = model.conv_head
        handle  = target.register_forward_hook(_save_fmap)

        img_orig = Image.open(file_path).convert('RGB')
        img_orig = _crop_to_face(img_orig)  # match the crop used in scoring

        # Cap size so the returned base64 stays reasonable
        max_side = 800
        w, h = img_orig.size
        if max(w, h) > max_side:
            scale    = max_side / max(w, h)
            img_orig = img_orig.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        tensor = _transform(img_orig).unsqueeze(0)   # [1, 3, 224, 224]

        model.zero_grad()
        # Forward pass WITHOUT no_grad — graph is required for backward
        outputs = model(tensor)

        handle.remove()

        fmap = feature_map[0]             # [1, C, H, W] still in graph
        fmap.retain_grad()                # allow .grad on a non-leaf tensor

        score = outputs[0, 0]             # class 0 = fake
        score.backward()

        if fmap.grad is None:
            return None

        # ── Compute Grad-CAM ─────────────────────────────────────────────
        grads = fmap.grad.detach().squeeze(0)      # [C, H, W]
        acts  = fmap.detach().squeeze(0)           # [C, H, W]

        weights = grads.mean(dim=(1, 2))           # global avg-pool → [C]
        cam     = (weights[:, None, None] * acts).sum(0)   # [H, W]
        cam     = torch.relu(cam).numpy()

        if cam.max() == 0:
            return None
        cam = cam / cam.max()            # normalise to [0, 1]

        # ── Build heatmap and blend ──────────────────────────────────────
        orig_w, orig_h = img_orig.size
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((orig_w, orig_h), Image.BILINEAR)
        cam_np  = np.array(cam_pil).astype(np.float32) / 255.0

        # "Hot" colormap: black → red → yellow → white
        r = np.clip(cam_np * 3.0,       0, 1)
        g = np.clip(cam_np * 3.0 - 1.0, 0, 1)
        b = np.clip(cam_np * 3.0 - 2.0, 0, 1)
        heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

        orig_np = np.array(img_orig)
        blended = (0.55 * orig_np + 0.45 * heatmap).clip(0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format='JPEG', quality=85)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception:
        return None