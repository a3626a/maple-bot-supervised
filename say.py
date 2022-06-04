# 특정 음성을 재생하는 say 함수를 제공한다.
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import threading

def say(text):
    def say_async():
        mp3_fp = BytesIO()
        tts = gTTS(text, lang='ko')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        sound = AudioSegment.from_file(mp3_fp, format="mp3")
        play(sound)
    threading.Thread(target=say_async).start()
