"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import (TextPreprocessor, TextEncoder, AdjacencyMatrixGraph, VanillaTextRank,
                                          EdgeListGraph, PositionBiasedTextRank)


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

    processed = TextPreprocessor(stop_words, punctuation)
    words = processed.preprocess_text(text)

    encoded = TextEncoder()
    tokens = encoded.encode(words)

    graph = AdjacencyMatrixGraph()
    graph.fill_from_tokens(tokens, 7)

    van_graph = VanillaTextRank(graph)
    van_graph.train()
    top10_adj = van_graph.get_top_keywords(10)
    decoded10_adj = encoded.decode(top10_adj)
    print(decoded10_adj)

    edge_list_graph = EdgeListGraph()
    edge_list_graph.fill_from_tokens(tokens, 7)

    van_edge_graph = VanillaTextRank(edge_list_graph)
    van_edge_graph.train()
    top10_edge = van_edge_graph.get_top_keywords(10)
    decoded10_edge = encoded.decode(top10_edge)
    print(decoded10_edge)

    position_graph1 = PositionBiasedTextRank(graph)
    position_graph1.train()
    top_10_adj2 = position_graph1.get_top_keywords(10)
    decoded10_adj2 = encoded.decode(top_10_adj2)
    print(decoded10_adj2)

    position_graph2 = PositionBiasedTextRank(edge_list_graph)
    position_graph2.train()
    top_10_edge2 = position_graph2.get_top_keywords(10)
    decoded10_edge2 = encoded.decode(top_10_edge2)
    print(decoded10_edge2)

    RESULT = decoded10_edge2
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
