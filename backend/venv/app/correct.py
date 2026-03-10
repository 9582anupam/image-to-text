from autocorrect import Speller

spell = Speller()

def correct_text(text):
    return spell(text)
