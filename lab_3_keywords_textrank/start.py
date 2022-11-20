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

    preprocessor = TextPreprocessor(stop_words, tuple(punctuation))
    clean_tokens = preprocessor.preprocess_text(text)

    text_encoder = TextEncoder()
    encoded_tokens = text_encoder.encode(clean_tokens)
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
    DECODED_TOP_10_VANILLA = text_encoder.decode(top_10_vanilla)
    print(DECODED_TOP_10_VANILLA)

    RESULT = DECODED_TOP_10_VANILLA
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
