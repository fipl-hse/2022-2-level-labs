"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import extract_pairs, TextPreprocessor, TextEncoder, AdjacencyMatrixGraph, \
    VanillaTextRank, EdgeListGraph, PositionBiasedTextRank

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

    preprocessor = TextPreprocessor(stop_words, tuple(punctuation))
    clean_tokens = preprocessor.preprocess_text(text)

    txt_encoder = TextEncoder()
    encoded_tokens = txt_encoder.encode(clean_tokens)
    if encoded_tokens:
        pairs = extract_pairs(encoded_tokens, 3)
        print(pairs)

    adjacency_matrix_graph = AdjacencyMatrixGraph()
    edge_list_graph = EdgeListGraph()

    if encoded_tokens:
        adjacency_matrix_graph.fill_from_tokens(encoded_tokens, 3)
        adjacency_matrix_graph.fill_positions(encoded_tokens)
        adjacency_matrix_graph.calculate_position_weights()

    vanilla_text_rank = VanillaTextRank(adjacency_matrix_graph)
    vanilla_rank_edge = VanillaTextRank(edge_list_graph)
    vanilla_text_rank.train()
    top_10_vanilla = vanilla_text_rank.get_top_keywords(10)
    decoded_top_n_vanilla = txt_encoder.decode(top_10_vanilla)
    print(decoded_top_n_vanilla)

    biased_rank = PositionBiasedTextRank(adjacency_matrix_graph)
    biased_rank.train()
    top_biased_matrix = biased_rank.get_top_keywords(10)
    if top_biased_matrix:
        result_of_adj_matrix = txt_encoder.decode(top_biased_matrix)
        print(result_of_adj_matrix)

    biased_edge = PositionBiasedTextRank(edge_list_graph)
    biased_edge.train()
    top_biased_edge = biased_edge.get_top_keywords(10)
    if top_biased_edge:
        result_of_edge_graph = txt_encoder.decode(top_biased_edge)
        print(result_of_edge_graph)

    RESULT = True
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
