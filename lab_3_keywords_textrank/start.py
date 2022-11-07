"""
TextRank keyword extraction starter
"""

from pathlib import Path
from lab_3_keywords_textrank.main import TextPreprocessor, TextEncoder, AdjacencyMatrixGraph, VanillaTextRank
from string import punctuation

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

    preprocessed_text = TextPreprocessor(stop_words, tuple(punctuation))
    tokens = preprocessed_text.preprocess_text(text)
    encoded_text = TextEncoder()
    encoded_tokens = encoded_text.encode(tokens)
    graph = AdjacencyMatrixGraph()
    graph.fill_from_tokens(encoded_tokens, 3)
    rank = VanillaTextRank(graph)
    rank.train()
    top = rank.get_top_keywords(10)
    decoded_text = encoded_text.decode(top)

    RESULT = decoded_text
    assert RESULT, 'Keywords are not extracted'
