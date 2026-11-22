import sys, wave, threading
import sounddevice as sd
import numpy as np
import re
import os

_cuda_libs = []
for _mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        _m = __import__(_mod, fromlist=["__file__"])
        _cuda_libs.append(os.path.dirname(_m.__file__))
    except Exception:
        pass
if _cuda_libs:
    os.environ["LD_LIBRARY_PATH"] = ":".join(_cuda_libs + [os.environ.get("LD_LIBRARY_PATH", "")])

from faster_whisper import WhisperModel   # must come AFTER the lines above
from piper import PiperVoice
from ask import ask
from store import init_db, embed_pending

SR = 16000
_whisper = WhisperModel("medium", device="cpu", compute_type="int8")
_voice = PiperVoice.load("en_US-amy-medium.onnx")


def listen():
    frames = []
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            callback=lambda indata, *_: frames.append(indata.copy()))
    print("🎤 Listening... press Enter when you're done speaking.")
    with stream:
        input()
    if not frames:
        return ""
    audio = np.concatenate(frames).flatten()
    segments, info = _whisper.transcribe(audio, beam_size=5)
    text = " ".join(s.text for s in segments).strip()
    print(f"   you ({info.language}): {text}")
    return text


def speak(text):
    path = "answer.wav"
    with wave.open(path, "wb") as wav:
        _voice.synthesize_wav(text, wav)
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    sd.play(pcm, rate)
    sd.wait()

def for_speech(text):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)          # [label](url) -> label
    text = re.sub(r"(?:https?://|www\.)\S+", "", text, flags=re.I)  # bare URLs
    text = re.sub(r"\S+@\S+\.\S+", "", text)                       # email addresses
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.M)           # "1." list markers
    # symbols TTS would read aloud; keep .,!?;:' for natural prosody
    text = re.sub(r"[*#_`>~|\[\](){}<>\"“”‘’«»/\\=+^&%$@—–-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

if __name__ == "__main__":
    init_db()
    embed_pending()

    question = listen()
    if not question:
        print("Didn't catch anything.")
        sys.exit()
    answer = ask(question)
    print(f"\n🤖 {answer}\n")  # screen: keep the formatting
    speak(for_speech(answer))  # speaker: cleaned, no symbols read aloud