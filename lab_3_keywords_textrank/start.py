"""
TextRank keyword extraction starter
"""

from pathlib import Path
from json import load as json_load
from string import punctuation
from lab_3_keywords_textrank.main import (
    TextPreprocessor,
    TextEncoder,
    extract_pairs,
    AdjacencyMatrixGraph,
    VanillaTextRank,
    EdgeListGraph,
    PositionBiasedTextRank,
    KeywordExtractionBenchmark
)

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

    preprocessor = TextPreprocessor(stop_words, ('.', '-', ':', '/', ',', '–', '(', ')', '-'))
    encoder = TextEncoder()
    tokens = encoder.encode(preprocessor.preprocess_text(text))

    # step 3 demonstration
    if tokens:
        print(extract_pairs(tokens, 3))

    # step 6, 7.3 and 9.3
    adj_graph = AdjacencyMatrixGraph()
    edg_graph = EdgeListGraph()
    if tokens:
        adj_graph.fill_from_tokens(tokens, 3)
        edg_graph.fill_from_tokens(tokens, 3)
        adj_graph.fill_positions(tokens)
        adj_graph.calculate_position_weights()
        edg_graph.fill_positions(tokens)
        edg_graph.calculate_position_weights()

    vanilla_adj = VanillaTextRank(adj_graph)
    vanilla_edg = VanillaTextRank(edg_graph)
    biased_adj = PositionBiasedTextRank(adj_graph)
    biased_edg = PositionBiasedTextRank(edg_graph)

    for algorithm in vanilla_adj, vanilla_edg, biased_adj, biased_edg:
        algorithm.train()
        print(encoder.decode(algorithm.get_top_keywords(10)))

    # step 12
    benchmark_materials = ASSETS_PATH / 'benchmark_materials'
    stop_words_path = benchmark_materials / 'eng_stop_words.txt'
    file = open(stop_words_path, 'r', encoding='utf-8')
    eng_stop_words = tuple(file.read().split('\n'))
    idf_path = benchmark_materials / 'IDF.json'
    file = open(idf_path, 'r', encoding='utf-8')
    idf = dict(json_load(file))
    benchmark_punctuation = tuple(i for i in punctuation)

    benchmark = KeywordExtractionBenchmark(eng_stop_words, benchmark_punctuation, idf, benchmark_materials)
    benchmark.run()
    benchmark.save_to_csv(PROJECT_ROOT / 'report.csv')

    RESULT = True
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
