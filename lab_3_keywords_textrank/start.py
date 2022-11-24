"""
TextRank keyword extraction starter
"""

from pathlib import Path
from main import (TextPreprocessor,
                  TextEncoder,
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

    punctuation = ('.', ',', ':', '-', '?', '!', '...', ';')
    preprocessing = TextPreprocessor(stop_words, punctuation)
    preprocessed_text = preprocessing.preprocess_text(text)

    encoding = TextEncoder()
    encoded_text = encoding.encode(preprocessed_text)

    graph = AdjacencyMatrixGraph()
    graph.fill_from_tokens(encoded_text, 10)

    rank = VanillaTextRank(graph)
    rank.train()
    top = rank.get_top_keywords(10)

    decoded_words = encoding.decode(top)

    edge = EdgeListGraph()
    edge.fill_from_tokens(encoded_text, 10)

    rank_edge = VanillaTextRank(edge)
    rank_edge.train()
    top_edge = rank_edge.get_top_keywords(10)

    biased_rank_graph = PositionBiasedTextRank(graph)
    biased_rank_graph.train()
    top_biased_graph = biased_rank_graph.get_top_keywords(10)

    biased_rank_edge = PositionBiasedTextRank(edge)
    biased_rank_edge.train()
    top_biased_edge = biased_rank_edge.get_top_keywords(10)

    print(top_biased_graph)
    print(top_biased_edge)

    RESULT = decoded_words
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'



