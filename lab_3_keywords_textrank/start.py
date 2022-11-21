"""
TextRank keyword extraction starter
"""
import string
from pathlib import Path
import json
from lab_3_keywords_textrank.main import (TextPreprocessor, TextEncoder, extract_pairs,
                                          AdjacencyMatrixGraph, VanillaTextRank, EdgeListGraph,
                                          PositionBiasedTextRank, KeywordExtractionBenchmark)

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

    RESULT = None
    PAIRS = None

    # mark 4
    preprocessed_text = TextPreprocessor(stop_words, tuple(string.punctuation)).preprocess_text(text)
    encoder = TextEncoder()
    tokens = encoder.encode(preprocessed_text)
    if tokens:
        PAIRS = extract_pairs(tokens, 3)
    print(PAIRS)

    # mark 6
    matrix_graph = AdjacencyMatrixGraph()
    if tokens:
        matrix_graph.fill_from_tokens(tokens, 3)
        matrix_graph.fill_positions(tokens)
    matrix_graph.calculate_position_weights()

    vanilla_matrix = VanillaTextRank(matrix_graph)
    vanilla_matrix.train()
    top_vanilla_matrix = vanilla_matrix.get_top_keywords(10)
    if top_vanilla_matrix:
        res1 = encoder.decode(top_vanilla_matrix)
        print(res1)

    edge_graph = EdgeListGraph()
    if tokens:
        edge_graph.fill_from_tokens(tokens, 3)
        edge_graph.fill_positions(tokens)
    edge_graph.calculate_position_weights()

    vanilla_edge = VanillaTextRank(edge_graph)
    vanilla_edge.train()
    top_vanilla_edge = vanilla_edge.get_top_keywords(10)
    if top_vanilla_edge:
        res2 = encoder.decode(top_vanilla_edge)
        print(res2)

    # mark 8
    biased_matrix = PositionBiasedTextRank(matrix_graph)
    biased_matrix.train()
    top_biased_matrix = biased_matrix.get_top_keywords(10)
    if top_biased_matrix:
        res3 = encoder.decode(top_biased_matrix)
        print(res3)

    biased_edge = PositionBiasedTextRank(edge_graph)
    biased_edge.train()
    top_biased_edge = biased_edge.get_top_keywords(10)
    if top_biased_edge:
        RESULT = encoder.decode(top_biased_edge)
        print(RESULT)

    # mark 10
    with open(ASSETS_PATH / 'benchmark_materials' / "eng_stop_words.txt", 'r', encoding='utf-8') as file:
        eng_stop_words = tuple(file.read().split())

    with open(ASSETS_PATH / 'benchmark_materials' / 'IDF.json', 'r', encoding='utf-8') as file:
        idf = json.load(file)

    benchmark = KeywordExtractionBenchmark(eng_stop_words, tuple(string.punctuation), idf,
                                           ASSETS_PATH / 'benchmark_materials')
    benchmark.run()
    benchmark.save_to_csv(PROJECT_ROOT / 'report.csv')

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
