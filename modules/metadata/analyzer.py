import os
import json
import subprocess

AI_SOFTWARE_SIGNATURES = [
    'stable diffusion', 'midjourney', 'dall-e', 'firefly',
    'runway', 'synthesia', 'deepfacelab', 'faceswap',
    'elevenlabs', 'murf', 'resemble', 'voicebox',
    'suno', 'udio', 'generator', 'ai generated',
    'adobe firefly', 'canva', 'imagemagick'
]

def analyze(file_path: str, media_type: str) -> float:
    try:
        if media_type == 'image':
            return _analyze_image(file_path)
        elif media_type == 'video':
            return _analyze_video(file_path)
        elif media_type == 'audio':
            return _analyze_audio(file_path)
        return 0.5
    except Exception:
        return 0.5

def _analyze_image(file_path: str) -> float:
    try:
        import exifread
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=True)

        # --- No EXIF at all ---
        if not tags:
            # Could be screenshot or AI generated
            # Check file format for screenshot clues
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.png':
                # PNG with no EXIF = likely screenshot or AI generated
                return 0.60
            else:
                # JPEG with no EXIF = strongly suspicious
                return 0.75

        # --- AI software signature (immediate high score) ---
        software = str(tags.get('Image Software', '')).lower()
        if any(sig in software for sig in AI_SOFTWARE_SIGNATURES):
            return 0.92

        # --- Score based on presence of authentic camera fields ---
        score = 0.5  # neutral start

        has_make      = 'Image Make' in tags
        has_model     = 'Image Model' in tags
        has_datetime  = 'EXIF DateTimeOriginal' in tags or 'Image DateTime' in tags
        has_makernote = any('makernote' in k.lower() for k in tags.keys())
        has_gps       = any('gps' in k.lower() for k in tags.keys())
        has_exposure  = 'EXIF ExposureTime' in tags
        has_fnumber   = 'EXIF FNumber' in tags
        has_iso       = 'EXIF ISOSpeedRatings' in tags

        # Each authentic camera field reduces suspicion
        if has_make:      score -= 0.08
        if has_model:     score -= 0.08
        if has_datetime:  score -= 0.08
        if has_makernote: score -= 0.12  # strongest signal of real camera
        if has_exposure:  score -= 0.06
        if has_fnumber:   score -= 0.06
        if has_iso:       score -= 0.06
        if has_gps:       score -= 0.04

        # If none of the camera fields present despite having some EXIF
        if not any([has_make, has_model, has_makernote, has_exposure]):
            score += 0.20

        return round(min(max(score, 0.05), 0.95), 4)

    except Exception:
        return 0.5

def _analyze_video(file_path: str) -> float:
    flags = []

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', '-show_streams', file_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return 0.4

        data = json.loads(result.stdout)
        fmt  = data.get('format', {})
        tags = {k.lower(): v.lower() for k, v in fmt.get('tags', {}).items()}

        encoder = tags.get('encoder', '') + tags.get('comment', '') + tags.get('software', '')
        if any(sig in encoder for sig in AI_SOFTWARE_SIGNATURES):
            flags.append(0.95)

        if 'creation_time' not in tags:
            flags.append(0.2)

        streams = data.get('streams', [])
        has_video = any(s.get('codec_type') == 'video' for s in streams)
        has_audio = any(s.get('codec_type') == 'audio' for s in streams)

        if has_video and not has_audio:
            flags.append(0.3)

    except Exception:
        return 0.4

    if not flags:
        return 0.2

    return round(min(sum(flags) / max(len(flags), 1), 0.95), 4)

def _analyze_audio(file_path: str) -> float:
    flags = []

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', '-show_streams', file_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return 0.4

        data = json.loads(result.stdout)
        fmt  = data.get('format', {})
        tags = {k.lower(): v.lower() for k, v in fmt.get('tags', {}).items()}

        combined = ' '.join(tags.values())
        if any(sig in combined for sig in AI_SOFTWARE_SIGNATURES):
            flags.append(0.95)

        encoder = tags.get('encoder', '') + tags.get('comment', '') + tags.get('software', '')
        if any(sig in encoder for sig in AI_SOFTWARE_SIGNATURES):
            flags.append(0.95)

        streams = data.get('streams', [])
        audio_streams = [s for s in streams if s.get('codec_type') == 'audio']

        if audio_streams:
            bit_rate = int(fmt.get('bit_rate', 0))
            if 0 < bit_rate < 32000:
                flags.append(0.5)

            sample_rate = int(audio_streams[0].get('sample_rate', 44100))
            if sample_rate == 22050:
                flags.append(0.4)

    except Exception:
        return 0.4

    if not flags:
        return 0.25

    return round(min(sum(flags) / max(len(flags), 1), 0.95), 4)
