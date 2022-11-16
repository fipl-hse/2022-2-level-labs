"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import (TextPreprocessor,
                                          TextEncoder,
                                          extract_pairs,
                                          AdjacencyMatrixGraph,
                                          VanillaTextRank, )

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

    preprocessor = TextPreprocessor(stop_words, tuple(punctuation))
    tokens = preprocessor.preprocess_text(text)
    encoder = TextEncoder()
    encoded_tokens = encoder.encode(tokens)
    if encoded_tokens:
        print(extract_pairs(encoded_tokens, 5))

    adjacency_matrix_graph = AdjacencyMatrixGraph()
    if encoded_tokens:
        adjacency_matrix_graph.fill_from_tokens(encoded_tokens, 4)

    vanilla_matrix = VanillaTextRank(adjacency_matrix_graph)
    vanilla_matrix.train()
    top_vanilla_matrix = vanilla_matrix.get_top_keywords(6)
    if top_vanilla_matrix:
        words = encoder.decode(top_vanilla_matrix)
        print(words)

    RESULT = top_vanilla_matrix
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
