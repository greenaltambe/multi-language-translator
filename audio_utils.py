import numpy as np
import sounddevice as sd  # recording audio
import soundfile as sf  # save .wav files
import scipy.signal  # apply filter
from config import SAMPLE_RATE


def preprocess_audio(audio, sample_rate=SAMPLE_RATE):
    sos = scipy.signal.butter(10, 100, "hp", fs=sample_rate, output="sos")
    filtered_audio = scipy.signal.sosfilt(sos, audio)
    return filtered_audio


def record_audio(duration=5, sample_rate=SAMPLE_RATE):
    audio = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    audio = np.squeeze(audio).astype(np.float32)
    sf.write("debug_input.wav", audio, sample_rate)
    return audio
