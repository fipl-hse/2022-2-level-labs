"""
TextRank keyword extraction starter
"""

from pathlib import Path
from lab_3_keywords_textrank.main import *


if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / 'article.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = tuple(file.read().split('\n'))

    RESULT = None
    punctuation = ('!', '"', '#', '$', '%', '&', '''''', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';',
                   '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~')
    raw_text = TextPreprocessor(stop_words=stop_words, punctuation=punctuation)
    preprocessed_text = raw_text.preprocess_text(text=text)
    encode_text = TextEncoder()
    coded_text = encode_text.encode(tokens=preprocessed_text)
    decoded_text = encode_text.decode(encoded_tokens=coded_text)
    print('tokens: ', preprocessed_text, '\n',
          'coded tokens: ', coded_text, '\n',
          'decoded tokens: ', decoded_text, '\n')
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
