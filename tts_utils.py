from gtts import gTTS
import pygame

def save_and_play_tts(text, lang, output_file):
    tts = gTTS(text=text, lang=lang[:2].lower(), slow=False)
    tts.save(output_file)
    pygame.mixer.init()
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
