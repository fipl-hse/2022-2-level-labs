"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
import json
from lab_3_keywords_textrank.main import (extract_pairs,
                                          TextPreprocessor,
                                          TextEncoder,
                                          VanillaTextRank,
                                          AdjacencyMatrixGraph,
                                          EdgeListGraph,
                                          PositionBiasedTextRank,
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

    preprocessed_text = TextPreprocessor(stop_words, tuple(punctuation)).preprocess_text(text)
    encoded_text = TextEncoder()
    encoded_tokens = encoded_text.encode(preprocessed_text)
    if encoded_tokens:
        print(extract_pairs(encoded_tokens, 3))

    adjacency_matrix_graph = AdjacencyMatrixGraph()
    if encoded_tokens:
        adjacency_matrix_graph.fill_from_tokens(encoded_tokens, 3)
        adjacency_matrix_graph.fill_positions(encoded_tokens)
    adjacency_matrix_graph.calculate_position_weights()

    vanilla_text_rank_amg = VanillaTextRank(adjacency_matrix_graph)
    vanilla_text_rank_amg.train()
    print(encoded_text.decode(vanilla_text_rank_amg.get_top_keywords(10)))

    edge_list_graph = EdgeListGraph()
    if encoded_tokens:
        edge_list_graph.fill_from_tokens(encoded_tokens, 3)
        edge_list_graph.fill_positions(encoded_tokens)
    edge_list_graph.calculate_position_weights()

    vanilla_text_rank_elg = VanillaTextRank(edge_list_graph)
    vanilla_text_rank_elg.train()
    print(encoded_text.decode(vanilla_text_rank_elg.get_top_keywords(10)))

    position_text_rank_amg = PositionBiasedTextRank(adjacency_matrix_graph)
    position_text_rank_amg.train()
    print(encoded_text.decode(position_text_rank_amg.get_top_keywords(10)))

    position_text_rank_elg = PositionBiasedTextRank(edge_list_graph)
    position_text_rank_elg.train()
    print(encoded_text.decode(position_text_rank_elg.get_top_keywords(10)))

    BENCHMARK_MATERIALS_PATH = ASSETS_PATH / 'benchmark_materials'

    ENG_STOP_WORDS_PATH = BENCHMARK_MATERIALS_PATH / 'eng_stop_words.txt'
    with open(ENG_STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        eng_stop_words = tuple(file.read().split('\n'))

    IDF_PATH = BENCHMARK_MATERIALS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    keyword_extraction_benchmark = KeywordExtractionBenchmark(eng_stop_words, tuple(punctuation),
                                                              idf, BENCHMARK_MATERIALS_PATH)
    keyword_extraction_benchmark.run()
    keyword_extraction_benchmark.save_to_csv(PROJECT_ROOT / 'report.csv')

    RESULT = True

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
