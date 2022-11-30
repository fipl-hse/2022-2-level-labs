"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
import json

from lab_3_keywords_textrank.main import (extract_pairs, TextEncoder, TextPreprocessor, VanillaTextRank,
                                          AdjacencyMatrixGraph, PositionBiasedTextRank, EdgeListGraph,
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

    if text:
        preprocessor = TextPreprocessor(stop_words, punctuation)
        encoder = TextEncoder()
        tokens = encoder.encode(preprocessor.preprocess_text(text))
        if tokens:
            print(extract_pairs(tokens, 3))

    adj_graph = AdjacencyMatrixGraph()
    if tokens:
        adj_graph.fill_from_tokens(tokens, 5)
    adj_text_rank = VanillaTextRank(adj_graph)
    adj_text_rank.train()
    adj_top_keywords = adj_text_rank.get_top_keywords(10)
    encoded_adj_top = encoder.decode(adj_top_keywords)
    print(encoded_adj_top)

    # step 7.3
    edge_list = EdgeListGraph()
    if tokens:
        edge_list.fill_from_tokens(tokens, 5)
    edge_text_rank = VanillaTextRank(edge_list)
    edge_text_rank.train()
    edge_top_keywords = edge_text_rank.get_top_keywords(10)
    edge_encoded_top = encoder.decode(edge_top_keywords)
    print(edge_encoded_top)

    # step 9.3: AdjacencyMatrixGraph
    if tokens:
        adj_graph.fill_positions(tokens)
    adj_graph.calculate_position_weights()
    adj_bias_text_rank = PositionBiasedTextRank(adj_graph)
    adj_bias_text_rank.train()
    adj_bias_top_keywords = adj_bias_text_rank.get_top_keywords(10)
    encoded_adj_bias_top = encoder.decode(adj_bias_top_keywords)
    print(encoded_adj_bias_top)

    # step 9.3: EdgeListGraph
    if tokens:
        edge_list.fill_positions(tokens)
    edge_list.calculate_position_weights()
    edge_bias_text_rank = PositionBiasedTextRank(edge_list)
    edge_bias_text_rank.train()
    edge_bias_top_keywords = edge_bias_text_rank.get_top_keywords(10)
    encoded_edge_bias_top = encoder.decode(edge_bias_top_keywords)
    print(encoded_edge_bias_top)

    # step 12
    BENCHMARK_MATERIALS_PATH = ASSETS_PATH / 'benchmark_materials'

    IDF_PATH = BENCHMARK_MATERIALS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    ENG_STOP_WORDS_PATH = BENCHMARK_MATERIALS_PATH / 'eng_stop_words.txt'
    with open(ENG_STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        eng_stop_words = tuple(file.read().split('\n'))

    benchmark = KeywordExtractionBenchmark(eng_stop_words, tuple(punctuation), idf, BENCHMARK_MATERIALS_PATH)
    benchmark.run()
    benchmark.save_to_csv(PROJECT_ROOT / 'report.csv')

    RESULT = benchmark.report
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
