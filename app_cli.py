import os
import torch
import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf
import scipy.signal
import warnings
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import whisper
from gtts import gTTS
import uuid

warnings.filterwarnings("ignore")

# Configuration
SAMPLE_RATE = 16000
MODEL_DIR = "./models"
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


def load_translation_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(
        model_name, cache_dir=MODEL_DIR
    )
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR)
    return model, tokenizer


def load_whisper_model():
    return whisper.load_model("small", download_root=MODEL_DIR)


def preprocess_audio(audio, sample_rate=SAMPLE_RATE):
    sos = scipy.signal.butter(10, 100, "hp", fs=sample_rate, output="sos")
    filtered_audio = scipy.signal.sosfilt(sos, audio)
    return filtered_audio


def record_audio(duration=5, sample_rate=SAMPLE_RATE):
    print("Recording audio...")
    audio = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    audio = np.squeeze(audio).astype(np.float32)
    sf.write("debug_input.wav", audio, sample_rate)
    print("Recording complete.")
    return audio


def clean_transcription(text):
    words = text.split()
    cleaned = []
    last_word = None
    for word in words:
        if word != last_word or len(cleaned) == 0:
            cleaned.append(word)
            last_word = word
    return " ".join(cleaned)


def save_and_play_tts(text, lang, output_file):
    try:
        tts = gTTS(text=text, lang=lang[:2].lower(), slow=False)
        tts.save(output_file)
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing audio: {e}")


def translate_text(text, src_lang, tgt_lang, model, tokenizer):
    tokenizer.src_lang = LANG_CODES[src_lang]
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[LANG_CODES[tgt_lang]],
        max_length=100,
    )
    translated_text = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]
    return translated_text or "Translation failed."


def speech_to_text(audio, src_lang, whisper_model):
    audio = preprocess_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # Ensure audio is float32 before converting to tensor
    audio = torch.from_numpy(audio).float().to(whisper_model.device)

    mel = whisper.log_mel_spectrogram(audio)

    whisper_lang = WHISPER_CODES.get(src_lang)
    if not whisper_lang:
        print(f"Language {src_lang} not supported by Whisper.")
        return None

    options = whisper.DecodingOptions(
        language=whisper_lang, fp16=False, task="transcribe"
    )
    result = whisper.decode(whisper_model, mel, options)

    if not result.text or len(result.text) < 3:
        print("Transcription failed.")
        return None
    return clean_transcription(result.text)


def main():
    pygame.mixer.init()
    print("Loading models, please wait...")
    translation_model, tokenizer = load_translation_model()
    whisper_model = load_whisper_model()
    print("Models loaded!")

    print("\nSupported languages:")
    for lang in LANG_CODES.keys():
        print(f"- {lang}")

    src_lang = input("\nEnter Source Language: ").strip()
    tgt_lang = input("Enter Target Language: ").strip()

    print("\nChoose Mode:")
    print("1. Text-to-Text")
    print("2. Speech-to-Text")
    print("3. Speech-to-Speech")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        text = input("Enter text to translate: ")
        translated = translate_text(
            text, src_lang, tgt_lang, translation_model, tokenizer
        )
        print(f"\nTranslated Text: {translated}")
        save_audio = input(
            "Do you want to save and play the translated audio? (y/n): "
        ).lower()
        if save_audio == "y":
            output_file = f"output_{tgt_lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
            save_and_play_tts(translated, tgt_lang, output_file)

    elif choice == "2":
        audio = record_audio()
        text = speech_to_text(audio, src_lang, whisper_model)
        if text:
            translated = translate_text(
                text, src_lang, tgt_lang, translation_model, tokenizer
            )
            print(f"\nTranscribed Text: {text}")
            print(f"Translated Text: {translated}")
            save_audio = input(
                "Do you want to save and play the translated audio? (y/n): "
            ).lower()
            if save_audio == "y":
                output_file = f"output_{tgt_lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
                save_and_play_tts(translated, tgt_lang, output_file)

    elif choice == "3":
        audio = record_audio()
        text = speech_to_text(audio, src_lang, whisper_model)
        if text:
            translated = translate_text(
                text, src_lang, tgt_lang, translation_model, tokenizer
            )
            print(f"\nTranscribed Text: {text}")
            print(f"Translated Text: {translated}")
            output_file = f"output_{tgt_lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
            save_and_play_tts(translated, tgt_lang, output_file)

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        pygame.mixer.quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        pygame.mixer.quit()
