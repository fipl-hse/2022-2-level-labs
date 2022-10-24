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
    pairs = extract_pairs(tokens, 3)
    print(pairs)

    # step 6
    adj_graph = AdjacencyMatrixGraph()
    adj_graph.fill_from_tokens(tokens, 3)
    vanilla_text_rank1 = VanillaTextRank(adj_graph)
    vanilla_text_rank1.train()
    top_decoded1 = encoder.decode(vanilla_text_rank1.get_top_keywords(10))
    print(top_decoded1)

    # step 7.3
    edg_graph = EdgeListGraph()
    edg_graph.fill_from_tokens(tokens, 3)
    vanilla_text_rank2 = VanillaTextRank(edg_graph)
    vanilla_text_rank2.train()
    top_decoded2 = encoder.decode(vanilla_text_rank2.get_top_keywords(10))
    print(top_decoded2)

    # step 9.3
    adj_graph.fill_positions(tokens)
    adj_graph.calculate_position_weights()
    position_biased1 = PositionBiasedTextRank(adj_graph)
    position_biased1.train()
    top_decoded3 = encoder.decode(position_biased1.get_top_keywords(10))
    print(top_decoded3)

    edg_graph.fill_positions(tokens)
    edg_graph.calculate_position_weights()
    position_biased1 = PositionBiasedTextRank(edg_graph)
    position_biased1.train()
    top_decoded4 = encoder.decode(position_biased1.get_top_keywords(10))
    print(top_decoded4)

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

    RESULT = top_decoded1, top_decoded2, top_decoded3, top_decoded4
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
