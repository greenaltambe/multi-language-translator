import os

SAMPLE_RATE = 16000  # For audio recording
MODEL_DIR = "./models"  # Directory to cache models
os.makedirs(MODEL_DIR, exist_ok=True)

LANG_CODES = {
    "English": "en_XX",
    "Hindi": "hi_IN",
    "Tamil": "ta_IN",
    "Arabic": "ar_AR",
    "Mandarin": "zh_CN",
}

WHISPER_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Arabic": "ar",
    "Mandarin": "zh",
}
