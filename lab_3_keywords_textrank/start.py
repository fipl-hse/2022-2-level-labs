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
    # python -m pytest -m mark6
    punctuation = ('!', '"', '#', '$', '%', '&', '''''', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';',
                   '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~')
    decoded = None
    text_preprocessor = TextPreprocessor(stop_words=stop_words, punctuation=punctuation)
    tokens = text_preprocessor.preprocess_text(text=text)

    text_encoder = TextEncoder()
    encoded_tokens = text_encoder.encode(tokens=tokens)

    if encoded_tokens:
        graph = AdjacencyMatrixGraph()
        graph.fill_from_tokens(encoded_tokens, window_length=3)

        vanilla = VanillaTextRank(graph)
        vanilla.train()
        keywords = vanilla.get_top_keywords(10)
        decoded = text_encoder.decode(encoded_tokens=keywords)

    RESULT = decoded
    print(RESULT)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
