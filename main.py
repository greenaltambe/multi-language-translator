import tkinter as tk
from gui import TranslationApp
import pygame


def main():
    try:
        pygame.mixer.init()
        root = tk.Tk()
        app = TranslationApp(root)
        root.mainloop()
    except KeyboardInterrupt:
        pygame.mixer.quit()
        exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
