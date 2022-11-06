"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import (
    TextPreprocessor,
    TextEncoder,
    extract_pairs,
    AdjacencyMatrixGraph,
    VanillaTextRank)

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

    text_preprocessor = TextPreprocessor(stop_words, tuple(punctuation))
    text_preprocessed = text_preprocessor.preprocess_text(text)
    encoder = TextEncoder()
    tokens = encoder.encode(text_preprocessed)

    PAIRS = None

    if tokens:
        PAIRS = extract_pairs(tokens, 3)
    print(PAIRS)

    adjacency_matrix_graph = AdjacencyMatrixGraph()

    if tokens:
        adjacency_matrix_graph.fill_from_tokens(tokens, 3)
        adjacency_matrix_graph.fill_positions(tokens)
        adjacency_matrix_graph.calculate_position_weights()
    vanilla_text_rank = VanillaTextRank(adjacency_matrix_graph)
    vanilla_text_rank.score_vertices()
    top_10_vanilla = vanilla_text_rank.get_top_keywords(10)
    DECODED_TOP_10_VANILLA = encoder.decode(top_10_vanilla)
    print(DECODED_TOP_10_VANILLA)

    RESULT = DECODED_TOP_10_VANILLA
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
