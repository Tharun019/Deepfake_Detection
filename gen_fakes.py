import subprocess, os
out = os.path.expanduser("~/Desktop/sample inputs/AUD/fake_audio")
clips = [
 ("en-US-AriaNeural", "Last weekend I finally cleaned out the garage. I found boxes of old cables, a broken bicycle pump, and a stack of magazines from years ago. It took the whole afternoon, but the space feels completely different now, and I might even park the car inside."),
 ("en-US-GuyNeural", "The meeting has been moved to Thursday at three in the afternoon. Please send your updates before then so that we can go through them together. If you cannot attend, let me know in advance and I will share the notes with you afterwards."),
 ("en-GB-RyanNeural", "When you are learning a new programming language, it helps to build something small every day. Start with a simple calculator, then a to-do list, and later a tiny game. Over a few weeks, the concepts begin to feel natural instead of confusing."),
 ("en-GB-SoniaNeural", "The train to the coast leaves every hour from platform four. The journey takes about ninety minutes, and the views along the way are lovely. Remember to buy a return ticket at the station, because the machines at the seaside are often busy."),
 ("en-IN-NeerjaNeural", "Monsoon season brings a very different mood to the city. The roads shine after the rain, the air smells fresh, and everyone carries an umbrella. A hot cup of tea and a window seat is honestly all you need on days like these."),
 ("en-AU-WilliamNeural", "To make a quick pasta sauce, heat some olive oil and add chopped garlic. Pour in a can of tomatoes, add salt and a pinch of sugar, and let it simmer for fifteen minutes. Stir in fresh basil at the end and serve it over hot pasta."),
]
for i,(voice,text) in enumerate(clips,1):
    f = os.path.join(out, "fake_edge_%02d_%s.mp3" % (i, voice))
    subprocess.run(["edge-tts","--voice",voice,"--text",text,"--write-media",f], check=True)
    print("made", f)
