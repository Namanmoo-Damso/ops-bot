from gtts import gTTS

# Korean sentences to convert
sentences = [
    ("응, 그래 그래. 요즘 날씨가 좋아서 산책도 다니고 그래.", "voice_1.mp3"),
    ("그러게 말이야. 오늘 아침에 시장 갔다왔는데 사람이 많더라고.", "voice_2.mp3"),
    ("점심은 된장찌개 끓여 먹었어. 그냥 그랬어.", "voice_3.mp3"),
    ("아, 그래? 뭐라고? 다시 한번 말해봐.", "voice_4.mp3"),
    ("요즘 텔레비전에서 드라마 하더라. 봤어?", "voice_5.mp3"),
]

for text, filename in sentences:
    tts = gTTS(text=text, lang='ko')
    tts.save(filename)
    print(f"Saved: {filename}")

print("\nAll files saved successfully!")
