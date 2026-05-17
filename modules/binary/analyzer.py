import os
import math
import struct
import numpy as np
from collections import Counter


def analyze(file_path: str, media_type: str) -> float:
    if media_type == 'image':
        return _analyze_image(file_path)
    elif media_type == 'video':
        return _analyze_video(file_path)
    elif media_type == 'audio':
        return _analyze_audio(file_path)
    return 0.5


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
            return 0.6
        for qt in qt_tables:
            if len(qt) < 65:
                continue
            table = list(qt[1:65])
            if len(set(table)) <= 3:
                flags += 2
            avg = sum(table) / len(table)
            if avg < 2 or avg > 200:
                flags += 1
            if table[0] > min(table[1:8]):
                flags += 1
        return round(min(flags * 0.2, 0.95), 4)
    except Exception:
        return 0.5


def _is_heic(file_path: str, raw: bytes) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.heic', '.heif'):
        return True
    # check ftyp box for heic/heif magic bytes
    if len(raw) >= 12:
        ftyp = raw[4:8]
        brand = raw[8:12]
        if ftyp == b'ftyp' and brand in (b'heic', b'heix', b'hevc', b'mif1'):
            return True
    return False


def _analyze_image(file_path: str) -> float:
    scores = []
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()

        # HEIC is Apple iPhone native format — treat as camera-authentic
        if _is_heic(file_path, raw):
            return 0.15

        entropy = _shannon_entropy(raw)
        scores.append(0.65 if (entropy < 6.5 or entropy > 7.95) else 0.3)
        scores.append(_byte_uniformity_score(raw))
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            scores.append(_jpeg_quantization_score(file_path))
        try:
            from PIL import Image
            img = Image.open(file_path)
            w, h = img.size
            ratio = len(raw) / (w * h + 1)
            scores.append(0.65 if (ratio < 0.3 or ratio > 5.0) else 0.3)
        except Exception:
            pass
        return round(sum(scores) / len(scores), 4) if scores else 0.5
    except Exception:
        return 0.5


def _analyze_video(file_path: str) -> float:
    flags = 0
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            sample = f.read(min(65536, size))
        if len(header) >= 8:
            if header[4:8] not in (b'ftyp', b'moov', b'mdat', b'free'):
                flags += 1
        if _shannon_entropy(sample) < 7.0:
            flags += 1
        if _byte_uniformity_score(sample) > 0.7:
            flags += 1
        return round(min(flags * 0.25, 0.95), 4)
    except Exception:
        return 0.5


def _analyze_audio(file_path: str) -> float:
    flags = 0
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
        elif ext == '.mp3':
            valid = (header[:3] == b'ID3' or
                     (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0))
            if not valid:
                flags += 1
        elif ext == '.flac':
            if header[:4] != b'fLaC':
                flags += 2
        if _shannon_entropy(sample) < 6.0:
            flags += 1
        if _byte_uniformity_score(sample) > 0.65:
            flags += 1
        return round(min(flags * 0.25, 0.95), 4)
    except Exception:
        return 0.5
