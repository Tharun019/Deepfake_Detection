import os
import math
import struct
import numpy as np
from collections import Counter


def analyze(file_path: str, media_type: str):
    if media_type == 'image':
        return _analyze_image(file_path)
    elif media_type == 'video':
        return _analyze_video(file_path)
    elif media_type == 'audio':
        return _analyze_audio(file_path)
    return 0.5, {}


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counter = Counter(data)
    total = len(data)
    return -sum((c / total) * math.log2(c / total)
                for c in counter.values() if c > 0)


def _byte_uniformity_score(data: bytes) -> float:
    if not data:
        return 0.5
    counts = np.array(
        [Counter(data).get(i, 0) for i in range(256)], dtype=float
    )
    expected = len(data) / 256
    chi = float(np.sum((counts - expected) ** 2 / (expected + 1e-10)))
    normalized = min(chi / 500000.0, 1.0)
    return round(1.0 - normalized, 4)


def _jpeg_quantization_score(file_path: str) -> float:
    flags = 0
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        i, qt_tables = 0, []
        while i < len(data) - 3:
            if data[i] == 0xFF and data[i + 1] == 0xDB:
                length = struct.unpack('>H', data[i + 2:i + 4])[0]
                qt_tables.append(data[i + 4: i + 2 + length])
                i += 2 + length
            else:
                i += 1
        if not qt_tables:
            return 0.3
        for qt in qt_tables:
            if len(qt) < 65:
                continue
            table = list(qt[1:65])
            if len(set(table)) <= 3:
                flags += 2
            avg = sum(table) / len(table)
            if avg < 2 or avg > 200:
                flags += 1
        return round(min(flags * 0.25, 0.95), 4)
    except Exception:
        return 0.5


def _is_heic(file_path: str, raw: bytes) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.heic', '.heif'):
        return True
    if len(raw) >= 12:
        ftyp = raw[4:8]
        brand = raw[8:12]
        if ftyp == b'ftyp' and brand in (b'heic', b'heix', b'hevc', b'mif1'):
            return True
    return False


def _analyze_image(file_path: str):
    scores = []
    features = {}

    try:
        with open(file_path, 'rb') as f:
            raw = f.read()

        if _is_heic(file_path, raw):
            features['heic_format'] = {
                'label': 'HEIC format — Apple iPhone native, camera-authentic',
                'contribution': 0.15,
                'direction': 'authentic'
            }
            return 0.15, features

        entropy = _shannon_entropy(raw)
        if entropy < 6.0 or entropy > 8.0:
            entropy_score = 0.65
            features['shannon_entropy'] = {
                'label': f'Shannon entropy {entropy:.3f} — outside normal camera range (6.0–8.0)',
                'contribution': 0.65,
                'direction': 'suspicious'
            }
        else:
            entropy_score = 0.30
            features['shannon_entropy'] = {
                'label': f'Shannon entropy {entropy:.3f} — within normal camera range',
                'contribution': 0.30,
                'direction': 'authentic'
            }
        scores.append(entropy_score)

        uniformity = _byte_uniformity_score(raw)
        if uniformity > 0.90:
            uniformity_contribution = 0.30
            uniformity_direction = 'suspicious'
        else:
            uniformity_contribution = min(uniformity, 0.30)
            uniformity_direction = 'authentic'
        features['byte_uniformity'] = {
            'label': f'Byte uniformity score: {uniformity:.3f}',
            'contribution': uniformity_contribution,
            'direction': uniformity_direction
        }
        scores.append(uniformity_contribution)

        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            qt_score = _jpeg_quantization_score(file_path)
            features['jpeg_quantization'] = {
                'label': f'JPEG quantization table score: {qt_score:.3f}',
                'contribution': qt_score * 0.5,
                'direction': 'suspicious' if qt_score > 0.6 else 'authentic'
            }
            scores.append(qt_score * 0.5)
        elif ext == '.png':
            features['png_no_qt'] = {
                'label': 'PNG format — no JPEG quantization artifacts (expected for lossless format)',
                'contribution': 0.30,
                'direction': 'neutral'
            }
            scores.append(0.30)
        try:
            from PIL import Image
            img = Image.open(file_path)
            w, h = img.size
            ratio = len(raw) / (w * h + 1)
            if ratio < 0.3 or ratio > 5.0:
                features['bytes_per_pixel'] = {
                    'label': f'Bytes-per-pixel ratio {ratio:.3f} — abnormal for camera image',
                    'contribution': 0.65,
                    'direction': 'suspicious'
                }
                scores.append(0.65)
            else:
                features['bytes_per_pixel'] = {
                    'label': f'Bytes-per-pixel ratio {ratio:.3f} — normal',
                    'contribution': 0.30,
                    'direction': 'authentic'
                }
                scores.append(0.30)
        except Exception:
            pass

        final_score = round(sum(scores) / len(scores), 4) if scores else 0.5
        return final_score, features

    except Exception:
        return 0.5, {}

def _analyze_audio(file_path: str):
    flags = 0
    features = {}

    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            sample = f.read(min(65536, size))

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.wav':
            if not (header[:4] == b'RIFF' and header[8:12] == b'WAVE'):
                flags += 2
                features['wav_header'] = {
                    'label': 'Invalid WAV header (RIFF/WAVE signature missing)',
                    'contribution': +0.50,
                    'direction': 'suspicious'
                }
            else:
                features['wav_header'] = {
                    'label': 'Valid WAV header',
                    'contribution': 0.0,
                    'direction': 'authentic'
                }
        elif ext == '.mp3':
            valid = (header[:3] == b'ID3' or
                     (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0))
            if not valid:
                flags += 1
                features['mp3_header'] = {
                    'label': 'Invalid MP3 header (ID3/sync bytes missing)',
                    'contribution': +0.25,
                    'direction': 'suspicious'
                }
            else:
                features['mp3_header'] = {
                    'label': 'Valid MP3 header',
                    'contribution': 0.0,
                    'direction': 'authentic'
                }
        elif ext == '.flac':
            if header[:4] != b'fLaC':
                flags += 2
                features['flac_header'] = {
                    'label': 'Invalid FLAC header (fLaC magic bytes missing)',
                    'contribution': +0.50,
                    'direction': 'suspicious'
                }
            else:
                features['flac_header'] = {
                    'label': 'Valid FLAC header',
                    'contribution': 0.0,
                    'direction': 'authentic'
                }

        entropy = _shannon_entropy(sample)
        if entropy < 6.0:
            flags += 1
            features['audio_entropy'] = {
                'label': f'Low entropy {entropy:.3f} — expected >6.0 for real audio',
                'contribution': +0.25,
                'direction': 'suspicious'
            }
        else:
            features['audio_entropy'] = {
                'label': f'Entropy {entropy:.3f} — normal for audio',
                'contribution': 0.0,
                'direction': 'authentic'
            }

        uniformity = _byte_uniformity_score(sample)
        if uniformity > 0.65:
            flags += 1
            features['audio_uniformity'] = {
                'label': f'High byte uniformity {uniformity:.3f} — suspicious for audio',
                'contribution': +0.25,
                'direction': 'suspicious'
            }
        else:
            features['audio_uniformity'] = {
                'label': f'Byte uniformity {uniformity:.3f} — normal',
                'contribution': 0.0,
                'direction': 'authentic'
            }

        return round(min(flags * 0.25, 0.95), 4), features

    except Exception:
        return 0.5, {}