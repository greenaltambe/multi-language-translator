import os
import tkinter as tk
from tkinter import ttk, messagebox
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
import threading
import queue
import uuid

warnings.filterwarnings("ignore")

# Configuration
SAMPLE_RATE = 16000  # For audio recording
MODEL_DIR = "./models"  # Directory to cache models
os.makedirs(MODEL_DIR, exist_ok=True)

# Supported languages
LANG_CODES = {
    "English": "en_XX",  # English
    "Hindi": "hi_IN",  # Hindi
    "Tamil": "ta_IN",  # Tamil
    "Arabic": "ar_AR",  # Arabic
    "Mandarin": "zh_CN",  # Mandarin
}

WHISPER_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Arabic": "ar",
    "Mandarin": "zh"
}

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Translation System")
        self.root.geometry("600x500")

        # Initialize models in a separate thread to prevent UI freeze
        self.translation_model = None
        self.tokenizer = None
        self.whisper_model = None
        self.model_loading = False
        self.audio_queue = queue.Queue()

        # GUI Elements
        self.create_widgets()

        # Start loading models in background
        self.load_models_thread()

    def create_widgets(self):
        # Language Selection
        tk.Label(self.root, text="Source Language:").pack(pady=5)
        self.src_lang = ttk.Combobox(self.root, values=list(LANG_CODES.keys()), state="readonly")
        self.src_lang.pack()
        self.src_lang.set("English")

        tk.Label(self.root, text="Target Language:").pack(pady=5)
        self.tgt_lang = ttk.Combobox(self.root, values=list(LANG_CODES.keys()), state="readonly")
        self.tgt_lang.pack()
        self.tgt_lang.set("Hindi")

        # Translation Modes
        tk.Button(self.root, text="Text-to-Text Translation", command=self.text_to_text).pack(pady=10)
        tk.Button(self.root, text="Speech-to-Text Translation", command=self.speech_to_text).pack(pady=10)
        tk.Button(self.root, text="Speech-to-Speech Translation", command=self.speech_to_speech).pack(pady=10)

        # Text Input
        tk.Label(self.root, text="Input Text:").pack(pady=5)
        self.input_text = tk.Text(self.root, height=5, width=50)
        self.input_text.pack()

        # Output Display
        tk.Label(self.root, text="Output:").pack(pady=5)
        self.output_text = tk.Text(self.root, height=5, width=50, state='disabled')
        self.output_text.pack()

        # Save Audio Checkbox
        self.save_audio_var = tk.BooleanVar()
        tk.Checkbutton(self.root, text="Save and Play Translated Audio", variable=self.save_audio_var).pack(pady=5)

        # Status Label
        self.status_label = tk.Label(self.root, text="Loading models...", fg="blue")
        self.status_label.pack(pady=10)

    def load_translation_model(self):
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.translation_model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=MODEL_DIR)
        self.tokenizer = MBart50Tokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR)

    def load_whisper_model(self):
        self.whisper_model = whisper.load_model("small", download_root=MODEL_DIR)

    def load_models_thread(self):
        def load():
            self.model_loading = True
            self.load_translation_model()
            self.load_whisper_model()
            self.model_loading = False
            self.root.after(0, self.update_status, "Ready")

        threading.Thread(target=load, daemon=True).start()

    def update_status(self, message):
        self.status_label.config(text=message)

    def preprocess_audio(self, audio, sample_rate=SAMPLE_RATE):
        sos = scipy.signal.butter(10, 100, 'hp', fs=sample_rate, output='sos')
        filtered_audio = scipy.signal.sosfilt(sos, audio)
        return filtered_audio

    def record_audio(self, duration=5, sample_rate=SAMPLE_RATE):
        self.update_status("Recording audio...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio).astype(np.float32)
        sf.write("debug_input.wav", audio, sample_rate)
        self.update_status("Recording complete")
        return audio

    def clean_transcription(self, text):
        words = text.split()
        cleaned = []
        last_word = None
        for word in words:
            if word != last_word or len(cleaned) == 0:
                cleaned.append(word)
                last_word = word
        return " ".join(cleaned)

    def save_tts(self, text, lang, output_file):
        try:
            tts = gTTS(text=text, lang=lang[:2].lower(), slow=False)
            tts.save(output_file)
            pygame.mixer.init()
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Error playing audio: {e}")

    def translate_text(self, text, src_lang, tgt_lang):
        if not self.translation_model or not self.tokenizer:
            self.update_status("Models not loaded yet")
            return "Translation failed."
        self.tokenizer.src_lang = LANG_CODES[src_lang]
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.translation_model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[LANG_CODES[tgt_lang]],
            max_length=100
        )
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        if not translated_text.strip():
            return "Translation failed."
        return translated_text

    def speech_to_text(self, audio, src_lang):
        if not self.whisper_model:
            self.update_status("Whisper model not loaded yet")
            return None
        audio = self.preprocess_audio(audio)
        audio = audio.astype(np.float32)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
        whisper_lang = WHISPER_CODES.get(src_lang)
        if not whisper_lang:
            self.root.after(0, messagebox.showerror, "Error", f"Language {src_lang} not supported by Whisper.")
            return None
        options = whisper.DecodingOptions(language=whisper_lang, fp16=False, task="transcribe")
        result = whisper.decode(self.whisper_model, mel, options)
        if not result.text or len(result.text) < 3:
            self.root.after(0, messagebox.showerror, "Error", "Transcription failed.")
            return None
        return self.clean_transcription(result.text)

    def text_to_text(self):
        if self.model_loading:
            messagebox.showinfo("Info", "Models are still loading. Please wait.")
            return
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter text to translate.")
            return
        src_lang = self.src_lang.get()
        tgt_lang = self.tgt_lang.get()
        translated = self.translate_text(text, src_lang, tgt_lang)
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, translated)
        self.output_text.config(state='disabled')
        if self.save_audio_var.get():
            output_file = f"output_{tgt_lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
            threading.Thread(target=self.save_tts, args=(translated, tgt_lang, output_file), daemon=True).start()

    def speech_to_text(self):
        if self.model_loading:
            messagebox.showinfo("Info", "Models are still loading. Please wait.")
            return
        src_lang = self.src_lang.get()
        tgt_lang = self.tgt_lang.get()
        audio = self.record_audio()
        text = self.speech_to_text(audio, src_lang)
        if text:
            translated = self.translate_text(text, src_lang, tgt_lang)
            self.output_text.config(state='normal')
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Transcribed: {text}\nTranslated: {translated}")
            self.output_text.config(state='disabled')

    def speech_to_speech(self):
        if self.model_loading:
            messagebox.showinfo("Info", "Models are still loading. Please wait.")
            return
        src_lang = self.src_lang.get()
        tgt_lang = self.tgt_lang.get()
        audio = self.record_audio()
        text = self.speech_to_text(audio, src_lang)
        if text:
            translated = self.translate_text(text, src_lang, tgt_lang)
            self.output_text.config(state='normal')
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Transcribed: {text}\nTranslated: {translated}")
            self.output_text.config(state='disabled')
            output_file = f"output_{tgt_lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
            threading.Thread(target=self.save_tts, args=(translated, tgt_lang, output_file), daemon=True).start()

def main():
    try:
        pygame.mixer.init()
        root = tk.Tk()
        app = TranslationApp(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting gracefully...")
        pygame.mixer.quit()
        exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        pygame.mixer.quit()

if __name__ == "__main__":
    main()