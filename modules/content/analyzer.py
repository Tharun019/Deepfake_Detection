import os
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
_image_model = None
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
