"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
import time
import json
from lab_3_keywords_textrank.main import extract_pairs, TextEncoder, \
    TextPreprocessor, VanillaTextRank, AdjacencyMatrixGraph, \
    PositionBiasedTextRank, EdgeListGraph, KeywordExtractionBenchmark

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

    # text preprocessing and pairs extraction
    processor = TextPreprocessor(stop_words, tuple(punctuation))
    encoder = TextEncoder()
    tokens = encoder.encode(processor.preprocess_text(text))
    if tokens:
        print(extract_pairs(tokens, 5))

    # extract key phrases with adjacency matrix graph
    start_time = time.time()
    adjacency_graph = AdjacencyMatrixGraph()
    if tokens:
        adjacency_graph.fill_from_tokens(tokens, 5)
    adjacency_unbiased_ranking = VanillaTextRank(adjacency_graph)
    adjacency_unbiased_ranking.train()
    print(encoder.decode(adjacency_unbiased_ranking.get_top_keywords(10)))
    finish_time = time.time()
    print(f'completed in {finish_time - start_time:.2f} seconds')
    print()

    # extract key phrases with list of edges graph
    start_time = time.time()
    edge_graph = EdgeListGraph()
    if tokens:
        edge_graph.fill_from_tokens(tokens, 5)
    edge_unbiased_ranking = VanillaTextRank(edge_graph)
    edge_unbiased_ranking.train()
    finish_time = time.time()
    print(encoder.decode(edge_unbiased_ranking.get_top_keywords(10)))
    print(f'completed in {finish_time - start_time:.2f} seconds')
    print()

    #  extract positionally biased key phrases with adjacency matrix graph
    start_time = time.time()
    if tokens:
        adjacency_graph.fill_positions(tokens)
    adjacency_graph.calculate_position_weights()
    adjacency_positional_ranking = PositionBiasedTextRank(adjacency_graph)
    adjacency_positional_ranking.train()
    print(encoder.decode(adjacency_positional_ranking.get_top_keywords(10)))
    finish_time = time.time()
    print(f'completed in {finish_time - start_time:.2f} seconds')
    print()

    # extract positionally biased key phrases with list of edges graph
    start_time = time.time()
    if tokens:
        edge_graph.fill_positions(tokens)
    edge_graph.calculate_position_weights()
    edge_positional_ranking = PositionBiasedTextRank(edge_graph)
    edge_positional_ranking.train()
    print(encoder.decode(edge_positional_ranking.get_top_keywords(10)))
    finish_time = time.time()
    print(f'completed in {finish_time - start_time:.2f} seconds')
    print()

    ENGLISH_STOP_WORDS_PATH = ASSETS_PATH / 'benchmark_materials' / 'eng_stop_words.txt'
    with open(ENGLISH_STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        english_stop_words = tuple(file.read().split('\n'))

    IDF_PATH = ASSETS_PATH / 'benchmark_materials' / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    MATERIALS_PATH = ASSETS_PATH / 'benchmark_materials'
    bench = KeywordExtractionBenchmark(english_stop_words, tuple(punctuation), idf, MATERIALS_PATH)
    bench.run()
    bench.save_to_csv(PROJECT_ROOT / 'report.csv')

    RESULT = bench.report
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
