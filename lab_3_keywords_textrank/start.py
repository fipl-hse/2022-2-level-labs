"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from json import load as json_load
from lab_3_keywords_textrank.main import (TextPreprocessor, TextEncoder, extract_pairs, AdjacencyMatrixGraph,
                                          EdgeListGraph, VanillaTextRank, PositionBiasedTextRank,
                                          KeywordExtractionBenchmark)

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

    PAIRS = None

    # 3
    text_preprocessed = TextPreprocessor(stop_words, tuple(punctuation)).preprocess_text(text)
    text_encoder = TextEncoder()
    tokens = text_encoder.encode(text_preprocessed)
    if tokens:
        PAIRS = extract_pairs(tokens, 3)
    print(PAIRS)

    # 6
    adjacency_matrix = AdjacencyMatrixGraph()
    if tokens:
        adjacency_matrix.fill_from_tokens(tokens, 3)
        adjacency_matrix.fill_positions(tokens)
        adjacency_matrix.calculate_position_weights()

    vanilla_rank = VanillaTextRank(adjacency_matrix)
    vanilla_rank.train()
    top_vanilla = vanilla_rank.get_top_keywords(10)
    print(top_vanilla_decode := text_encoder.decode(top_vanilla))

    # 7.3
    edge_graph = EdgeListGraph()
    if tokens:
        edge_graph.fill_from_tokens(tokens, 3)
        edge_graph.fill_positions(tokens)
        edge_graph.calculate_position_weights()

    vanilla_rank = VanillaTextRank(edge_graph)
    vanilla_rank.train()
    top_vanilla = vanilla_rank.get_top_keywords(10)
    print(top_vanilla_decode := text_encoder.decode(top_vanilla))

    # 9.3
    biased_rank_edge = PositionBiasedTextRank(edge_graph)
    biased_rank_edge.train()
    top_biased_edge = biased_rank_edge.get_top_keywords(10)
    print(top_biased_decode_edge := text_encoder.decode(top_biased_edge))

    biased_rank_adj = PositionBiasedTextRank(adjacency_matrix)
    biased_rank_adj.train()
    top_biased_adj = biased_rank_adj.get_top_keywords(10)
    print(top_biased_decode_adj := text_encoder.decode(top_biased_adj))

    # 12
    with open(ASSETS_PATH / 'benchmark_materials' / "eng_stop_words.txt", 'r', encoding='utf-8') as file:
        english_stop = tuple(file.read().split())

    with open(ASSETS_PATH / 'benchmark_materials' / 'IDF.json', 'r', encoding='utf-8') as file:
        idf = dict(json_load(file))

    if idf:
        benchmark = KeywordExtractionBenchmark(english_stop, tuple(punctuation), idf,
                                               ASSETS_PATH / 'benchmark_materials')
        benchmark.run()
        benchmark.save_to_csv(PROJECT_ROOT / 'report.csv')

    RESULT = PAIRS, top_vanilla_decode, top_biased_decode_edge, top_biased_decode_adj
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
