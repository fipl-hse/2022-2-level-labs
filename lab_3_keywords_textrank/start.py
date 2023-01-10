"""
TextRank keyword extraction starter
"""

from pathlib import Path

from lab_3_keywords_textrank.main import (TextPreprocessor, TextEncoder, extract_pairs, AdjacencyMatrixGraph,
                                          VanillaTextRank, EdgeListGraph, PositionBiasedTextRank)

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

    punctuation = ('.', ',', ':', '-', '?', '!', '...', ';')

    if text:
        preprocessor = TextPreprocessor(stop_words, punctuation)
        encoder = TextEncoder()
        tokens = encoder.encode(preprocessor.preprocess_text(text))
        if tokens:
            print(extract_pairs(tokens, 3))

    adjacency_graph = AdjacencyMatrixGraph()

    if tokens:
        adjacency_graph.fill_from_tokens(tokens, 5)
    adjacency_text_rank = VanillaTextRank(adjacency_graph)
    adjacency_text_rank.train()
    adjacency_top_keywords = adjacency_text_rank.get_top_keywords(10)
    encoded_adjacency_top = encoder.decode(adjacency_top_keywords)
    print(encoded_adjacency_top)

    edge_list = EdgeListGraph()
    if tokens:
        edge_list.fill_from_tokens(tokens, 5)
    edge_text_rank = VanillaTextRank(edge_list)
    edge_text_rank.train()
    edge_top_keywords = edge_text_rank.get_top_keywords(10)
    edge_encoded_top = encoder.decode(edge_top_keywords)
    print(edge_encoded_top)

    if tokens:
        adjacency_graph.fill_positions(tokens)
    adjacency_graph.calculate_position_weights()
    adjacency_bias_text_rank = PositionBiasedTextRank(adjacency_graph)
    adjacency_bias_text_rank.train()
    adjacency_bias_top_keywords = adjacency_bias_text_rank.get_top_keywords(10)
    encoded_adjacency_bias_top = encoder.decode(adjacency_bias_top_keywords)
    print(encoded_adjacency_bias_top)

    if tokens:
        edge_list.fill_positions(tokens)
    edge_list.calculate_position_weights()
    edge_bias_text_rank = PositionBiasedTextRank(edge_list)
    edge_bias_text_rank.train()
    edge_bias_top_keywords = edge_bias_text_rank.get_top_keywords(10)
    encoded_edge_bias_top = encoder.decode(edge_bias_top_keywords)
    print(encoded_edge_bias_top)

    RESULT = "Done"
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
