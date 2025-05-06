from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import whisper
from config import MODEL_DIR


def load_translation_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(
        model_name, cache_dir=MODEL_DIR
    ) # performs the translation
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR) # preprocesses the text for the model
    return model, tokenizer


def load_whisper_model():
    return whisper.load_model("small", download_root=MODEL_DIR)
