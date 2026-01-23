from gtts import gTTS

# Korean sentences to convert
sentences = [
    ("응, 그래 그래. 요즘 허리가 좀 아프긴 해도 잘 지내고 있어.", "voice_1.mp3"),
    ("아이고, 그러게 말이야. 나이 먹으니까 여기저기 안 아픈 데가 없어.", "voice_2.mp3"),
    ("점심은 된장찌개 끓여 먹었어. 맛있었어.", "voice_3.mp3"),
    ("아, 그래? 뭐라고? 다시 한번 말해봐.", "voice_4.mp3"),
    ("좀 우울해... 아무것도 하기 싫어.", "voice_5.mp3"),
]

for text, filename in sentences:
    tts = gTTS(text=text, lang='ko')
    tts.save(filename)
    print(f"Saved: {filename}")

print("\nAll files saved successfully!")
