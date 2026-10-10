import sys, json
from modules.content.audio_analyzer import analyze_audio

for f in sys.argv[1:]:
    print(f)
    print(json.dumps(analyze_audio(f), indent=1))
