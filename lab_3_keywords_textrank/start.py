"""
TextRank keyword extraction starter
"""

from string import punctuation
from pathlib import Path
from lab_3_keywords_textrank.main import TextPreprocessor, TextEncoder, AdjacencyMatrixGraph,\
    VanillaTextRank, extract_pairs, EdgeListGraph, PositionBiasedTextRank

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
    new_text = preprocessor.preprocess_text(text)
    encoder = TextEncoder()
    encoded_text = encoder.encode(new_text)

    if encoded_text:
        PAIRS = extract_pairs(encoded_text, 3)
        print(PAIRS)

    adj_matrix = AdjacencyMatrixGraph()
    if encoded_text:
        adj_matrix.fill_from_tokens(encoded_text, 3)
        adj_matrix.fill_positions(encoded_text)
        adj_matrix.get_position_weights()
    vanilla = VanillaTextRank(adj_matrix)
    vanilla.train()
    top_list_adj = vanilla.get_top_keywords(10)
    TOP_10_VANILLA_ADJ = encoder.decode(top_list_adj)
    print(TOP_10_VANILLA_ADJ)

    edge_matrix = EdgeListGraph()
    if encoded_text:
        edge_matrix.fill_from_tokens(encoded_text, 3)
        edge_matrix.fill_positions(encoded_text)
        edge_matrix.get_position_weights()
    vanilla = VanillaTextRank(edge_matrix)
    vanilla.train()
    top_list_edge = vanilla.get_top_keywords(10)
    TOP_10_VANILLA_EDGE = encoder.decode(top_list_edge)
    print(TOP_10_VANILLA_EDGE)

    adj_biased = PositionBiasedTextRank(adj_matrix)
    adj_biased.train()
    top_adj_biased = adj_biased.get_top_keywords(10)
    TOP_10_ADJ_BIASED = encoder.decode(top_adj_biased)
    print(TOP_10_ADJ_BIASED)

    edge_biased = PositionBiasedTextRank(edge_matrix)
    edge_biased.train()
    top_edge_biased = edge_biased.get_top_keywords(10)
    TOP_10_EDGE_BIASED = encoder.decode(top_edge_biased)
    print(TOP_10_EDGE_BIASED)

    RESULT = TOP_10_ADJ_BIASED


    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
