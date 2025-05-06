import whisper
from config import LANG_CODES, WHISPER_CODES
from audio_utils import preprocess_audio


def clean_transcription(text):
    words = text.split()
    cleaned = []
    last_word = None
    for word in words:
        if word != last_word or len(cleaned) == 0:
            cleaned.append(word)
            last_word = word
    return " ".join(cleaned)


def translate_text(model, tokenizer, text, src_lang, tgt_lang):
    tokenizer.src_lang = LANG_CODES[src_lang]
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[LANG_CODES[tgt_lang]],
        max_length=100
    )
    translated_text = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]
    return translated_text


def transcribe_audio(model, audio, src_lang):
    audio = preprocess_audio(audio)
    audio = audio.astype("float32")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    whisper_lang = WHISPER_CODES.get(src_lang)
    if not whisper_lang:
        return None
    options = whisper.DecodingOptions(
        language=whisper_lang, fp16=False, task="transcribe"
    )
    result = whisper.decode(model, mel, options)
    if not result.text or len(result.text) < 3:
        return None
    return clean_transcription(result.text)
