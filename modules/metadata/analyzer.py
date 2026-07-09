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


def analyze(file_path: str, media_type: str):
    try:
        if media_type == 'image':
            return _analyze_image(file_path)
        elif media_type == 'video':
            return _analyze_video(file_path)
        elif media_type == 'audio':
            return _analyze_audio(file_path)
        return 0.5, {}
    except Exception:
        return 0.5, {}


def _analyze_image(file_path: str):
    try:
        import exifread
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=True)

        features = {}

        if not tags:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.png':
                features['no_exif_png'] = {
                    'label': 'No EXIF — PNG file',
                    'contribution': +0.60,
                    'direction': 'suspicious'
                }
                return 0.60, features
            else:
                features['no_exif_jpeg'] = {
                    'label': 'No EXIF — JPEG file (strongly suspicious)',
                    'contribution': +0.75,
                    'direction': 'suspicious'
                }
                return 0.75, features

        software = str(tags.get('Image Software', '')).lower()
        if any(sig in software for sig in AI_SOFTWARE_SIGNATURES):
            features['ai_software_tag'] = {
                'label': f'AI software signature detected: {software.strip()}',
                'contribution': +0.92,
                'direction': 'suspicious'
            }
            return 0.92, features

        score = 0.5
        has_make      = 'Image Make' in tags
        has_model     = 'Image Model' in tags
        has_datetime  = 'EXIF DateTimeOriginal' in tags or 'Image DateTime' in tags
        has_makernote = any('makernote' in k.lower() for k in tags.keys())
        has_gps       = any('gps' in k.lower() for k in tags.keys())
        has_exposure  = 'EXIF ExposureTime' in tags
        has_fnumber   = 'EXIF FNumber' in tags
        has_iso       = 'EXIF ISOSpeedRatings' in tags

        if has_make:
            score -= 0.08
            features['camera_make'] = {'label': 'Camera make present', 'contribution': -0.08, 'direction': 'authentic'}
        else:
            features['camera_make'] = {'label': 'Camera make missing', 'contribution': 0.0, 'direction': 'neutral'}

        if has_model:
            score -= 0.08
            features['camera_model'] = {'label': 'Camera model present', 'contribution': -0.08, 'direction': 'authentic'}
        else:
            features['camera_model'] = {'label': 'Camera model missing', 'contribution': 0.0, 'direction': 'neutral'}

        if has_datetime:
            score -= 0.08
            features['datetime'] = {'label': 'Original datetime present', 'contribution': -0.08, 'direction': 'authentic'}
        else:
            features['datetime'] = {'label': 'Original datetime missing', 'contribution': 0.0, 'direction': 'neutral'}

        if has_makernote:
            score -= 0.12
            features['makernote'] = {'label': 'MakerNote present (strong camera signal)', 'contribution': -0.12, 'direction': 'authentic'}
        else:
            features['makernote'] = {'label': 'MakerNote absent (strong suspicion signal)', 'contribution': 0.0, 'direction': 'neutral'}

        if has_exposure:
            score -= 0.06
            features['exposure'] = {'label': 'Exposure time present', 'contribution': -0.06, 'direction': 'authentic'}
        else:
            features['exposure'] = {'label': 'Exposure time missing', 'contribution': 0.0, 'direction': 'neutral'}

        if has_fnumber:
            score -= 0.06
            features['fnumber'] = {'label': 'F-number present', 'contribution': -0.06, 'direction': 'authentic'}
        else:
            features['fnumber'] = {'label': 'F-number missing', 'contribution': 0.0, 'direction': 'neutral'}

        if has_iso:
            score -= 0.06
            features['iso'] = {'label': 'ISO speed present', 'contribution': -0.06, 'direction': 'authentic'}
        else:
            features['iso'] = {'label': 'ISO speed missing', 'contribution': 0.0, 'direction': 'neutral'}

        if has_gps:
            score -= 0.04
            features['gps'] = {'label': 'GPS coordinates present', 'contribution': -0.04, 'direction': 'authentic'}
        else:
            features['gps'] = {'label': 'GPS data absent', 'contribution': 0.0, 'direction': 'neutral'}

        if not any([has_make, has_model, has_makernote, has_exposure]):
            score += 0.20
            features['no_camera_fields'] = {
                'label': 'No core camera fields found despite EXIF present',
                'contribution': +0.20,
                'direction': 'suspicious'
            }

        return round(min(max(score, 0.05), 0.95), 4), features

    except Exception:
        return 0.5, {}


def _analyze_video(file_path: str):
    flags = []
    features = {}

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', '-show_streams', file_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return 0.4, {}

        data = json.loads(result.stdout)
        fmt  = data.get('format', {})
        tags = {k.lower(): v.lower() for k, v in fmt.get('tags', {}).items()}

        encoder = tags.get('encoder', '') + tags.get('comment', '') + tags.get('software', '')
        if any(sig in encoder for sig in AI_SOFTWARE_SIGNATURES):
            flags.append(0.95)
            features['ai_encoder_tag'] = {'label': 'AI software encoder tag detected', 'contribution': +0.95, 'direction': 'suspicious'}

        if 'creation_time' not in tags:
            flags.append(0.2)
            features['no_creation_time'] = {'label': 'No creation timestamp in container', 'contribution': +0.20, 'direction': 'suspicious'}
        else:
            features['creation_time'] = {'label': 'Creation timestamp present', 'contribution': -0.10, 'direction': 'authentic'}

        streams = data.get('streams', [])
        has_video = any(s.get('codec_type') == 'video' for s in streams)
        has_audio = any(s.get('codec_type') == 'audio' for s in streams)

        if has_video and not has_audio:
            flags.append(0.3)
            features['video_no_audio'] = {'label': 'Video stream with no audio track', 'contribution': +0.30, 'direction': 'suspicious'}
        else:
            features['av_streams'] = {'label': 'Both audio and video streams present', 'contribution': -0.05, 'direction': 'authentic'}

    except Exception:
        return 0.4, {}

    if not flags:
        return 0.2, features

    return round(min(sum(flags) / max(len(flags), 1), 0.95), 4), features


def _analyze_audio(file_path: str):
    flags = []
    features = {}

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', '-show_streams', file_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return 0.4, {}

        data = json.loads(result.stdout)
        fmt  = data.get('format', {})
        tags = {k.lower(): v.lower() for k, v in fmt.get('tags', {}).items()}

        combined = ' '.join(tags.values())
        if any(sig in combined for sig in AI_SOFTWARE_SIGNATURES):
            flags.append(0.95)
            features['ai_tag_combined'] = {'label': 'AI software tag in metadata fields', 'contribution': +0.95, 'direction': 'suspicious'}

        encoder = tags.get('encoder', '') + tags.get('comment', '') + tags.get('software', '')
        if any(sig in encoder for sig in AI_SOFTWARE_SIGNATURES):
            flags.append(0.95)
            features['ai_encoder'] = {'label': 'AI software encoder signature', 'contribution': +0.95, 'direction': 'suspicious'}

        streams = data.get('streams', [])
        audio_streams = [s for s in streams if s.get('codec_type') == 'audio']

        if audio_streams:
            bit_rate = int(fmt.get('bit_rate', 0))
            if 0 < bit_rate < 32000:
                flags.append(0.5)
                features['low_bitrate'] = {'label': f'Suspiciously low bitrate: {bit_rate}bps', 'contribution': +0.50, 'direction': 'suspicious'}
            else:
                features['bitrate'] = {'label': f'Bitrate normal: {bit_rate}bps', 'contribution': 0.0, 'direction': 'neutral'}

            sample_rate = int(audio_streams[0].get('sample_rate', 44100))
            if sample_rate == 22050:
                flags.append(0.4)
                features['low_sample_rate'] = {'label': 'Sample rate 22050Hz — common in TTS output', 'contribution': +0.40, 'direction': 'suspicious'}
            else:
                features['sample_rate'] = {'label': f'Sample rate normal: {sample_rate}Hz', 'contribution': 0.0, 'direction': 'neutral'}

    except Exception:
        return 0.4, {}

    if not flags:
        return 0.25, features

    return round(min(sum(flags) / max(len(flags), 1), 0.95), 4), features