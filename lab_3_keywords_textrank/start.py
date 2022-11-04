"""
TextRank keyword extraction starter
"""

from pathlib import Path
from lab_3_keywords_textrank.main import (TextPreprocessor, TextEncoder, extract_pairs,
                                          AdjacencyMatrixGraph)


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

    prep = TextPreprocessor(stop_words, ('!', ',', '(', ')', ':', '/', '-', '.'))
    tokens = prep.preprocess_text(text)
    print(tokens)

    encoder = TextEncoder()
    encoded = encoder.encode(tokens)
    print(encoded)
    print(encoder.decode(encoded))
    if encoded:
        print(extract_pairs(encoded, 3))
    graph = AdjacencyMatrixGraph()
    print(graph.add_edge(1001, 1005))

    # RESULT = None
    # # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    # assert RESULT, 'Keywords are not extracted'
