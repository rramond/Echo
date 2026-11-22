# whisper_test.py
import sounddevice as sd, numpy as np, threading

import os

# Put the venv's pip-installed CUDA libs on the loader path before CTranslate2 loads.
_cuda_libs = []
for _mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        _m = __import__(_mod, fromlist=["__file__"])
        _cuda_libs.append(os.path.dirname(_m.__file__))
    except Exception:
        pass
if _cuda_libs:
    os.environ["LD_LIBRARY_PATH"] = ":".join(_cuda_libs + [os.environ.get("LD_LIBRARY_PATH", "")])

from faster_whisper import WhisperModel

SR = 16000
model = WhisperModel("medium", device="cuda", compute_type="float16")

frames = []
stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda indata, *_: frames.append(indata.copy()))
print("🎤 Speak, then press Enter to stop.")
with stream:
    input()
audio = np.concatenate(frames).flatten()

segments, info = model.transcribe(audio, beam_size=5)
print(f"[detected: {info.language}]", " ".join(s.text for s in segments).strip())