"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import TextPreprocessor, TextEncoder, extract_pairs, AdjacencyMatrixGraph, \
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

    # step 3
    preprocessor = TextPreprocessor(stop_words, tuple(punctuation))
    encoder = TextEncoder()
    tokens = encoder.encode(preprocessor.preprocess_text(text))

    if tokens:
        print(extract_pairs(tokens, 3))

    # step 6
    adj_graph = AdjacencyMatrixGraph()
    if tokens:
        adj_graph.fill_from_tokens(tokens, 3)
    vanilla_rank_adj = VanillaTextRank(adj_graph)
    vanilla_rank_adj.train()
    top_keywords_adj = vanilla_rank_adj.get_top_keywords(10)
    print(encoder.decode(top_keywords_adj))

    # step 7.3
    edg_graph = EdgeListGraph()
    if tokens:
        edg_graph.fill_from_tokens(tokens, 3)
    vanilla_rank_edg = VanillaTextRank(edg_graph)
    vanilla_rank_edg.train()
    top_keywords_edg = vanilla_rank_edg.get_top_keywords(10)
    RESULT = encoder.decode(top_keywords_edg)
    print(RESULT)

    # step 9.3
    if tokens:
        adj_graph.fill_positions(tokens)
    adj_graph.calculate_position_weights()
    adj_biased = PositionBiasedTextRank(adj_graph)
    adj_biased.train()
    top_keywords_biased_adj = adj_biased.get_top_keywords(10)
    print(encoder.decode(top_keywords_biased_adj))

    if tokens:
        edg_graph.fill_positions(tokens)
    edg_graph.calculate_position_weights()
    edg_biased = PositionBiasedTextRank(edg_graph)
    edg_biased.train()
    top_keywords_biased_edg = edg_biased.get_top_keywords(10)
    RESULT = encoder.decode(top_keywords_biased_edg)
    print(RESULT)

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
