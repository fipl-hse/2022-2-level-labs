"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation

from lab_3_keywords_textrank.main import extract_pairs, TextPreprocessor, \
    TextEncoder, AdjacencyMatrixGraph, VanillaTextRank, EdgeListGraph, PositionBiasedTextRank
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
    preprocessed_text = preprocessor.preprocess_text(text)
    encoder = TextEncoder()
    ids_tokens = encoder.encode(preprocessed_text)
    token_ids = encoder.decode(ids_tokens)

    if token_ids:
        pairs = extract_pairs(ids_tokens, 3)
        print(pairs)

    adjacency_matrix = AdjacencyMatrixGraph()
    edge_graph = EdgeListGraph()
    if token_ids:
        adjacency_matrix.fill_from_tokens(ids_tokens, 3)
        adjacency_matrix.fill_positions(ids_tokens)
        adjacency_matrix.calculate_position_weights()

    vanilla_rank = VanillaTextRank(adjacency_matrix)
    vanilla_rank_edge = VanillaTextRank(edge_graph)
    vanilla_rank.train()
    top_keywords_adj = vanilla_rank.get_top_keywords(10)
    decoded_10_vanilla_rank = encoder.decode(top_keywords_adj)
    print(decoded_10_vanilla_rank)
    # vanilla_rank_edge = VanillaTextRank(edge_graph)
    # top_ten_edge = vanilla_rank_edge.get_top_keywords(10)
    # DECODED_TEN_EDGE = encoder.decode(top_ten_edge)
    biased_rank = PositionBiasedTextRank(adjacency_matrix)
    biased_rank.train()
    top_biased_matrix = biased_rank.get_top_keywords(10)
    if top_biased_matrix:
        result_of_adj_matrix = encoder.decode(top_biased_matrix)
        print(result_of_adj_matrix)

    biased_edge = PositionBiasedTextRank(edge_graph)
    biased_edge.train()
    top_biased_edge = biased_edge.get_top_keywords(10)
    if top_biased_edge:
        result_of_edge_graph = encoder.decode(top_biased_edge)
        print(result_of_edge_graph)

    RESULT = decoded_10_vanilla_rank
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
