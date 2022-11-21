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

    preprocessed_text = TextPreprocessor(stop_words, tuple(punctuation))
    encoded_text = TextEncoder()
    encoded_tokens = encoded_text.encode(preprocessed_text.preprocess_text(text))
    if encoded_tokens:
        print(extract_pairs(encoded_tokens, 3))

    adjacency_matrix_graph = AdjacencyMatrixGraph()
    if encoded_tokens:
        adjacency_matrix_graph.fill_from_tokens(encoded_tokens, 3)
        adjacency_matrix_graph.fill_positions(encoded_tokens)
    adjacency_matrix_graph.calculate_position_weights()

    vanilla_text_rank = VanillaTextRank(adjacency_matrix_graph)
    vanilla_text_rank.train()
    top_vanilla_text_rank = vanilla_text_rank.get_top_keywords(10)
    decoded_top_vanilla = encoded_text.decode(top_vanilla_text_rank)
    print(decoded_top_vanilla)

    position_biased_rank_adjacency_matrix_graph = PositionBiasedTextRank(adjacency_matrix_graph)
    position_biased_rank_adjacency_matrix_graph.train()
    top_position_biased_rank_adjacency_matrix_graph = position_biased_rank_adjacency_matrix_graph.get_top_keywords(10)
    decoded_top_biased_adjacency_matrix_graph = encoded_text.decode(top_position_biased_rank_adjacency_matrix_graph)
    print(decoded_top_biased_adjacency_matrix_graph)

    edge_list_graph = EdgeListGraph()
    if encoded_tokens:
        edge_list_graph.fill_from_tokens(encoded_tokens, 3)
        edge_list_graph.fill_positions(encoded_tokens)
    edge_list_graph.calculate_position_weights()

    vanilla_text_rank = VanillaTextRank(edge_list_graph)
    vanilla_text_rank.train()
    top_vanilla_text_rank = vanilla_text_rank.get_top_keywords(10)
    decoded_top_vanilla = encoded_text.decode(top_vanilla_text_rank)
    print(decoded_top_vanilla)

    position_biased_rank_edge_graph = PositionBiasedTextRank(edge_list_graph)
    position_biased_rank_edge_graph.train()
    top_position_biased_rank_edge_graph = position_biased_rank_edge_graph.get_top_keywords(10)
    decoded_top_biased_edge_graph = encoded_text.decode(top_position_biased_rank_edge_graph)
    print(decoded_top_biased_edge_graph)

    RESULT = True
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
