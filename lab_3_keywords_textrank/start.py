"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import extract_pairs, TextPreprocessor, TextEncoder, AdjacencyMatrixGraph, \
    VanillaTextRank


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

    # step 3
    preprocessor = TextPreprocessor(stop_words, tuple(punctuation))
    preprocessed_text = preprocessor.preprocess_text(text)
    encoder = TextEncoder()
    tokens = encoder.encode(preprocessed_text)
    if tokens:
        print(extract_pairs(tokens, 3))

    # step 6
    adjacency_matrix_graph = AdjacencyMatrixGraph()
    if tokens:
        adjacency_matrix_graph.fill_from_tokens(tokens, 3)
        adjacency_matrix_graph.fill_positions(tokens)
        adjacency_matrix_graph.calculate_position_weights()
    vanilla_rank_adjacency = VanillaTextRank(adjacency_matrix_graph)
    vanilla_rank_adjacency.train()
    print(encoder.decode(vanilla_rank_adjacency.get_top_keywords(10)))

    RESULT = None
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
