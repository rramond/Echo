import wave
from piper import PiperVoice

voice = PiperVoice.load("en_US-lessac-medium.onnx")
with wave.open("test.wav", "wb") as wav:
    voice.synthesize_wav("If you can hear this, Piper works.", wav)
print("wrote test.wav")