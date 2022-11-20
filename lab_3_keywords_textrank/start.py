"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import (TextPreprocessor,
                                          TextEncoder,
                                          extract_pairs,
                                          AdjacencyMatrixGraph,
                                          VanillaTextRank,
                                          EdgeListGraph,
                                          PositionBiasedTextRank)

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
        print(extract_pairs(encoded_tokens, 3))

    adjacency_matrix_graph = AdjacencyMatrixGraph()
    if encoded_tokens:
        adjacency_matrix_graph.fill_from_tokens(encoded_tokens, 3)
        adjacency_matrix_graph.fill_positions(encoded_tokens)
    adjacency_matrix_graph.calculate_position_weights()

    vanilla_matrix = VanillaTextRank(adjacency_matrix_graph)
    vanilla_matrix.train()
    top_vanilla_matrix = vanilla_matrix.get_top_keywords(10)
    if top_vanilla_matrix:
        top_words = encoder.decode(top_vanilla_matrix)
        print(top_words)

    edge_list = EdgeListGraph()
    if encoded_tokens:
        edge_list.fill_from_tokens(encoded_tokens, 3)
        edge_list.fill_positions(encoded_tokens)
    edge_list.calculate_position_weights()

    vanilla_rank = VanillaTextRank(edge_list)
    vanilla_rank.train()
    top_vanilla_edge = vanilla_rank.get_top_keywords(10)
    if top_vanilla_edge:
        top_words2 = encoder.decode(top_vanilla_edge)
        print(top_words2)

    biased_matrix = PositionBiasedTextRank(adjacency_matrix_graph)
    biased_matrix.train()
    top_biased_matrix = biased_matrix.get_top_keywords(10)
    if top_biased_matrix:
        top_words3 = encoder.decode(top_biased_matrix)
        print(top_words3)

    biased_edge = PositionBiasedTextRank(edge_list)
    biased_edge.train()
    top_biased_edge = biased_edge.get_top_keywords(10)
    top_biased_decoded_edge = encoder.decode(top_biased_edge)
    print(top_biased_decoded_edge)

    RESULT = top_biased_decoded_edge
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
