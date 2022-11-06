"""
TextRank keyword extraction starter
"""
import string
from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import (TextPreprocessor, TextEncoder, extract_pairs,
                                          AdjacencyMatrixGraph, VanillaTextRank)

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
    pairs = None
    preprocessor = TextPreprocessor(stop_words, tuple(string.punctuation))
    tokens = preprocessor.preprocess_text(text)
    encoder = TextEncoder()
    numbers = encoder.encode(tokens)
    graph = AdjacencyMatrixGraph()
    if numbers:
        graph.fill_from_tokens(numbers, 5)
    rank = VanillaTextRank(graph)
    rank.train()
    top = rank.get_top_keywords(10)
    RESULT = encoder.decode(top)
    print(RESULT)

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
