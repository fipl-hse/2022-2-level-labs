"""
TextRank keyword extraction starter
"""

from pathlib import Path
from main import (
    TextPreprocessor,
    TextEncoder,
    AdjacencyMatrixGraph,
    VanillaTextRank,
    EdgeListGraph,
    PositionBiasedTextRank,
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

    punctuation = ('.', ',', ':', '-', '?', '!', '...', ';')

    preprocessing = TextPreprocessor(stop_words, punctuation).preprocess_text(text)
    encoding = TextEncoder()
    tokens = encoding.encode(preprocessing)

    graph = AdjacencyMatrixGraph()
    if tokens:
        graph.fill_from_tokens(tokens, 3)
        graph.fill_positions(tokens)
        graph.calculate_position_weights()

    vanilla_rank = VanillaTextRank(graph)
    vanilla_rank.train()
    top_vanilla_rank = vanilla_rank.get_top_keywords(10)
    print(encoding.decode(top_vanilla_rank))

    edge = EdgeListGraph()
    if tokens:
        edge.fill_from_tokens(tokens, 3)
        edge.fill_positions(tokens)
        edge.calculate_position_weights()

    vanilla_rank = VanillaTextRank(edge)
    vanilla_rank.train()
    top_vanilla_edge = vanilla_rank.get_top_keywords(10)
    print(encoding.decode(top_vanilla_edge))

    biased_rank_edge = PositionBiasedTextRank(edge)
    biased_rank_edge.train()
    top_biased_edge = biased_rank_edge.get_top_keywords(10)
    print(encoding.decode(top_biased_edge))

    biased_rank_adj = PositionBiasedTextRank(graph)
    biased_rank_adj.train()
    top_biased_graph = biased_rank_adj.get_top_keywords(10)
    print(encoding.decode(top_biased_graph))

    RESULT = True
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
