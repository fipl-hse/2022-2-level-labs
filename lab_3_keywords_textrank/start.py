"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
import json
from lab_3_keywords_textrank.main import extract_pairs, TextPreprocessor, TextEncoder, AdjacencyMatrixGraph, \
    VanillaTextRank, EdgeListGraph,  PositionBiasedTextRank, KeywordExtractionBenchmark

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
    tokens = encoder.encode(preprocessed_text)
    if tokens:
        print(extract_pairs(tokens, 3))

    # step 6, working with VTR through AdjacencyMatrixGraph
    adjacency_matrix_graph = AdjacencyMatrixGraph()
    if tokens:
        adjacency_matrix_graph.fill_from_tokens(tokens, 3)
    adjacency_ranking = VanillaTextRank(adjacency_matrix_graph)
    adjacency_ranking.train()
    print(encoder.decode(adjacency_ranking.get_top_keywords(10)))

    # step 7, working with VTR through EdgeListGraph
    edge_list_graph = EdgeListGraph()
    if tokens:
        edge_list_graph.fill_from_tokens(tokens, 3)
    edge_ranking = VanillaTextRank(edge_list_graph)
    edge_ranking.train()
    print(encoder.decode(edge_ranking.get_top_keywords(10)))

    # step 9, working with PTR(PositionTextRank) through AdjacencyMatrixGraph and
    # step 7, working with VTR through
    position_rank_adjacency = PositionBiasedTextRank(adjacency_matrix_graph)
    position_rank_edge = PositionBiasedTextRank(edge_list_graph)
    for position_rank in position_rank_adjacency, position_rank_edge:
        position_rank.train()
        print(encoder.decode(position_rank.get_top_keywords(10)))

    # step 12
    BENCHMARK_PATH = ASSETS_PATH / 'benchmark_materials'

    ENG_STOP_WORDS_PATH = BENCHMARK_PATH / 'eng_stop_words.txt'
    with open(ENG_STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        eng_stop_words = tuple(file.read().split('\n'))

    IDF_PATH = BENCHMARK_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = dict(json.load(file))

    REPORT_PATH = PROJECT_ROOT / 'report.csv'
    benchmark = KeywordExtractionBenchmark(eng_stop_words, tuple(punctuation), idf, BENCHMARK_PATH)
    benchmark.run()
    benchmark.save_to_csv(REPORT_PATH)

    RESULT = benchmark.report
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
