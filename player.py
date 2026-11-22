import wave, numpy as np, sounddevice as sd

with wave.open("test.wav", "rb") as wav:
    rate = wav.getframerate()
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

sd.play(pcm, rate)
sd.wait()