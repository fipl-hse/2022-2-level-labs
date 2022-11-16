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
    EdgeListGraph,
    VanillaTextRank,
    PositionBiasedTextRank
)

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
    adjacency_matrix_graph = AdjacencyMatrixGraph()
    edge_graph = EdgeListGraph()
    PAIRS = None

    if tokens:
        PAIRS = extract_pairs(tokens, 3)
    print(PAIRS)

    if tokens:
        adjacency_matrix_graph.fill_from_tokens(tokens, 3)
        adjacency_matrix_graph.fill_positions(tokens)
        adjacency_matrix_graph.calculate_position_weights()
    vanilla_rank_adj = VanillaTextRank(adjacency_matrix_graph)
    vanilla_rank_adj.train()
    top_10_vanilla_adj = vanilla_rank_adj.get_top_keywords(10)
    DECODED_TOP_10_VANILLA_ADJ = encoder.decode(top_10_vanilla_adj)
    print(DECODED_TOP_10_VANILLA_ADJ)

    if tokens:
        edge_graph.fill_from_tokens(tokens, 3)
        edge_graph.fill_positions(tokens)
        edge_graph.calculate_position_weights()
    vanilla_rank_edge = VanillaTextRank(edge_graph)
    vanilla_rank_edge.train()
    top_10_vanilla_edge = vanilla_rank_edge.get_top_keywords(10)
    DECODED_TOP_10_VANILLA_EDGE = encoder.decode(top_10_vanilla_edge)
    print(DECODED_TOP_10_VANILLA_EDGE)

    biased_rank_adj = PositionBiasedTextRank(adjacency_matrix_graph)
    biased_rank_adj.train()
    top_10_biased_adj = biased_rank_adj.get_top_keywords(10)
    DECODED_TOP_10_BIASED_ADJ = encoder.decode(top_10_biased_adj)
    print(DECODED_TOP_10_BIASED_ADJ)

    biased_rank_edge = PositionBiasedTextRank(edge_graph)
    biased_rank_edge.train()
    top_10_biased_edge = biased_rank_edge.get_top_keywords(10)
    DECODED_TOP_10_BIASED_EDGE = encoder.decode(top_10_biased_edge)
    print(DECODED_TOP_10_BIASED_EDGE)

    RESULT = DECODED_TOP_10_BIASED_ADJ
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
