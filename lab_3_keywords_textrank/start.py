"""
TextRank keyword extraction starter
"""

from pathlib import Path
from lab_3_keywords_textrank.main import (TextPreprocessor, TextEncoder, extract_pairs,
                                          AdjacencyMatrixGraph, VanillaTextRank, EdgeListGraph,
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

    prep = TextPreprocessor(stop_words, ('!', ',', '(', ')', ':', '/', '-', '.'))
    tokens = prep.preprocess_text(text)

    encoder = TextEncoder()
    encoded = encoder.encode(tokens)
    # print(encoded)
    decoded = encoder.decode(encoded)
    # print(decoded)

    if encoded:
        pairs = extract_pairs(encoded, 3)

    graph = AdjacencyMatrixGraph()
    graph.fill_from_tokens(encoded, 3)
    graph.fill_positions(encoded)

    vanilla_graph = VanillaTextRank(graph)
    # if encoded:

    vanilla_graph.train()
    top_vanilla_graph = vanilla_graph.get_top_keywords(10)
    top_10_vanilla_graph = encoder.decode(top_vanilla_graph)
    print(top_10_vanilla_graph)

    edge_list_graph = EdgeListGraph()
    edge_list_graph.fill_from_tokens(encoded, 3)
    edge_list_graph.fill_positions(encoded)
    edge_list_graph.calculate_position_weights()

    vanilla_edge_graph = VanillaTextRank(edge_list_graph)
    vanilla_edge_graph.train()
    top_vanilla_edge_graph = vanilla_edge_graph.get_top_keywords(10)
    top_10_vanilla_edge_graph = encoder.decode(top_vanilla_edge_graph)
    print(top_10_vanilla_edge_graph)

    biased_rank = PositionBiasedTextRank(graph)
    biased_rank.train()
    top_biased_rank = biased_rank.get_top_keywords(10)
    top_10_biased_rank = encoder.decode(top_biased_rank)
    print(top_10_biased_rank)

    # RESULT = top_10_vanilla_edge_graph
    # # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    # assert RESULT, 'Keywords are not extracted'
