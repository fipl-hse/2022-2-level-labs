"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation

from lab_3_keywords_textrank.main import extract_pairs, TextPreprocessor, \
    TextEncoder, AdjacencyMatrixGraph, VanillaTextRank
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
    # edge_graph = EdgeListGraph()
    if token_ids:
        adjacency_matrix.fill_from_tokens(ids_tokens, 3)
        # adjacency_matrix.is_incidental(1000, 1235)
        # adjacency_matrix.get_vertices()
        # adjacency_matrix.calculate_inout_score(1235)
    #     adjacency_matrix.fill_positions(ids_tokens)
    # adjacency_matrix.calculate_position_weights()

    vanilla_rank = VanillaTextRank(adjacency_matrix)
    # vanilla_rank_edge = VanillaTextRank(edge_graph)
    vanilla_rank.train()
    top_keywords_adj = vanilla_rank.get_top_keywords(10)
    DECODED_TOP_10_VANILLA = encoder.decode(top_keywords_adj)
    print(DECODED_TOP_10_VANILLA)

    RESULT = DECODED_TOP_10_VANILLA
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
