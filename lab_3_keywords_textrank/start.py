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
    ASSETS_PATH = PROJECT_ROOT / "assets"

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / "article.txt"
    with open(TARGET_TEXT_PATH, "r", encoding="utf8") as file:
        text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / "stop_words.txt"
    with open(STOP_WORDS_PATH, "r", encoding="utf8") as file:
        stop_words = tuple(file.read().split('\n'))

    PREPROCESSOR = TextPreprocessor(stop_words, tuple(punctuation))
    ENCODER = TextEncoder()
    TOKENS = ENCODER.encode(PREPROCESSOR.preprocess_text(text))
    if TOKENS:
        print(extract_pairs(TOKENS, 5))
        print()

    #  extract key phrases with adjacency matrix graph
    ADJACENCY_GRAPH = AdjacencyMatrixGraph()
    if TOKENS:
        ADJACENCY_GRAPH.fill_from_tokens(TOKENS, 5)
    ADJACENCY_UNBIASED_RANKING = VanillaTextRank(ADJACENCY_GRAPH)
    START_TIME = time.monotonic()
    ADJACENCY_UNBIASED_RANKING.train()
    FINISH_TIME = time.monotonic()
    print(ENCODER.decode(ADJACENCY_UNBIASED_RANKING.get_top_keywords(10)))
    print(f"trained in {FINISH_TIME - START_TIME:.5f} seconds")
    print()

    #  extract key phrases with list of edges graph
    EDGE_GRAPH = EdgeListGraph()
    if TOKENS:
        EDGE_GRAPH.fill_from_tokens(TOKENS, 5)
    EDGE_UNBIASED_RANKING = VanillaTextRank(EDGE_GRAPH)
    START_TIME = time.monotonic()
    EDGE_UNBIASED_RANKING.train()
    FINISH_TIME = time.monotonic()
    print(ENCODER.decode(EDGE_UNBIASED_RANKING.get_top_keywords(10)))
    print(f"trained in {FINISH_TIME - START_TIME:.5f} seconds")
    print()

    #  extract positionally biased key phrases with adjacency matrix graph
    if TOKENS:
        ADJACENCY_GRAPH.fill_positions(TOKENS)
    ADJACENCY_GRAPH.calculate_position_weights()
    ADJACENCY_POSITIONAL_RANKING = PositionBiasedTextRank(ADJACENCY_GRAPH)
    START_TIME = time.monotonic()
    ADJACENCY_POSITIONAL_RANKING.train()
    FINISH_TIME = time.monotonic()
    print(ENCODER.decode(ADJACENCY_POSITIONAL_RANKING.get_top_keywords(10)))
    print(f"trained in {FINISH_TIME - START_TIME:.5f} seconds")
    print()

    #  extract positionally biased key phrases with list of edges graph
    if TOKENS:
        EDGE_GRAPH.fill_positions(TOKENS)
    EDGE_GRAPH.calculate_position_weights()
    EDGE_POSITIONAL_RANKING = PositionBiasedTextRank(EDGE_GRAPH)
    START_TIME = time.monotonic()
    EDGE_POSITIONAL_RANKING.train()
    FINISH_TIME = time.monotonic()
    print(ENCODER.decode(EDGE_POSITIONAL_RANKING.get_top_keywords(10)))
    print(f"trained in {FINISH_TIME - START_TIME:.5f} seconds")
    print()

    ENGLISH_STOP_WORDS_PATH = ASSETS_PATH / "benchmark_materials" / "eng_stop_words.txt"
    with open(ENGLISH_STOP_WORDS_PATH, "r", encoding="utf8") as file:
        ENGLISH_STOP_WORDS = tuple(file.read().split("\n"))

    IDF_PATH = ASSETS_PATH / "benchmark_materials" / "IDF.json"
    with open(IDF_PATH, "r", encoding="utf8") as file:
        IDF = json.load(file)

    MATERIALS_PATH = ASSETS_PATH / "benchmark_materials"
    BENCH = KeywordExtractionBenchmark(ENGLISH_STOP_WORDS, tuple(punctuation), IDF, MATERIALS_PATH)
    BENCH.run()
    BENCH.save_to_csv(PROJECT_ROOT / "report.csv")
    BENCH.report
    RESULT = True

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
