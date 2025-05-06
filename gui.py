import tkinter as tk
from tkinter import ttk, messagebox
import threading
import uuid
import pygame

from model_loader import load_translation_model, load_whisper_model
from audio_utils import record_audio
from translator import translate_text, transcribe_audio
from tts_utils import save_and_play_tts
from config import LANG_CODES


class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Translation System")
        self.root.geometry("600x500")

        self.translation_model = None
        self.tokenizer = None
        self.whisper_model = None
        self.model_loading = False

        self.create_widgets()
        self.load_models_thread()

    def create_widgets(self):
        tk.Label(self.root, text="Source Language:").pack(pady=5)
        self.src_lang = ttk.Combobox(
            self.root, values=list(LANG_CODES.keys()), state="readonly"
        )
        self.src_lang.pack()
        self.src_lang.set("English")

        tk.Label(self.root, text="Target Language:").pack(pady=5)
        self.tgt_lang = ttk.Combobox(
            self.root, values=list(LANG_CODES.keys()), state="readonly"
        )
        self.tgt_lang.pack()
        self.tgt_lang.set("Hindi")

        tk.Button(
            self.root, text="Text-to-Text Translation", command=self.text_to_text
        ).pack(pady=10)
        tk.Button(
            self.root,
            text="Speech-to-Text Translation",
            command=self.speech_to_text_mode,
        ).pack(pady=10)
        tk.Button(
            self.root,
            text="Speech-to-Speech Translation",
            command=self.speech_to_speech_mode,
        ).pack(pady=10)

        tk.Label(self.root, text="Input Text:").pack(pady=5)
        self.input_text = tk.Text(self.root, height=5, width=50)
        self.input_text.pack()

        tk.Label(self.root, text="Output:").pack(pady=5)
        self.output_text = tk.Text(self.root, height=5, width=50, state="disabled")
        self.output_text.pack()

        self.save_audio_var = tk.BooleanVar()
        tk.Checkbutton(
            self.root,
            text="Save and Play Translated Audio",
            variable=self.save_audio_var,
        ).pack(pady=5)

        self.status_label = tk.Label(self.root, text="Loading models...", fg="blue")
        self.status_label.pack(pady=10)

    def load_models_thread(self):
        def load():
            self.model_loading = True
            self.translation_model, self.tokenizer = load_translation_model()
            self.whisper_model = load_whisper_model()
            self.model_loading = False
            self.update_status("Ready")

        threading.Thread(target=load, daemon=True).start()

    def update_status(self, message):
        self.status_label.config(text=message)

    def text_to_text(self):
        if self.model_loading:
            messagebox.showinfo("Info", "Models are still loading. Please wait.")
            return
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter text.")
            return
        src_lang = self.src_lang.get()
        tgt_lang = self.tgt_lang.get()
        translated = translate_text(
            self.translation_model, self.tokenizer, text, src_lang, tgt_lang
        )
        self.display_output(translated)
        if self.save_audio_var.get():
            output_file = f"output_{tgt_lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
            threading.Thread(
                target=save_and_play_tts,
                args=(translated, tgt_lang, output_file),
                daemon=True,
            ).start()

    def speech_to_text_mode(self):
        if self.model_loading:
            messagebox.showinfo("Info", "Models are still loading. Please wait.")
            return
        src_lang = self.src_lang.get()
        tgt_lang = self.tgt_lang.get()
        audio = record_audio()
        text = transcribe_audio(self.whisper_model, audio, src_lang)
        if text:
            # Set transcribed text into input_text box
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, text)

            translated = translate_text(
                self.translation_model, self.tokenizer, text, src_lang, tgt_lang
            )
            self.display_output(translated)

    def speech_to_speech_mode(self):
        if self.model_loading:
            messagebox.showinfo("Info", "Models are still loading. Please wait.")
            return
        src_lang = self.src_lang.get()
        tgt_lang = self.tgt_lang.get()
        audio = record_audio()
        text = transcribe_audio(self.whisper_model, audio, src_lang)
        if text:
            # Set transcribed text into input_text box
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, text)

            translated = translate_text(
                self.translation_model, self.tokenizer, text, src_lang, tgt_lang
            )
            self.display_output(translated)

            if self.save_audio_var.get():
                output_file = f"output_{tgt_lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
                threading.Thread(
                    target=save_and_play_tts,
                    args=(translated, tgt_lang, output_file),
                    daemon=True,
                ).start()

    def display_output(self, text):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled")
