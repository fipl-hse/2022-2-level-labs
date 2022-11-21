"""
TextRank keyword extraction starter
"""
import json
from pathlib import Path
from string import punctuation
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

    BENCHMARK_MATERIALS_PATH = ASSETS_PATH / 'benchmark_materials'

    IDF_PATH = BENCHMARK_MATERIALS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    ENGLISH_STOP_WORDS_PATH = BENCHMARK_MATERIALS_PATH / 'eng_stop_words.txt'
    with open(ENGLISH_STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        eng_stop_words = tuple(file.read().split('\n'))

    # mark 4: extract pairs from text
    preprocessor = TextPreprocessor(stop_words, tuple(punctuation))
    encoder = TextEncoder()
    tokens = preprocessor.preprocess_text(text)
    encoded_tokens = encoder.encode(tokens)
    if encoded_tokens:
        print(extract_pairs(encoded_tokens, 5))

    #   mark 6: extract keywords with AdjacencyMatrixGraph using VanillaTextRank
    # adjacency_graph = AdjacencyMatrixGraph()
    # if encoded_tokens:
    #     adjacency_graph.fill_from_tokens(encoded_tokens, 5)
    # text_rank_adj = VanillaTextRank(adjacency_graph)
    # text_rank_adj.train()
    # top_keywords_adj = text_rank_adj.get_top_keywords(10)
    # print(encoded_top_adj := encoder.decode(top_keywords_adj))

    #   mark 8: extract keywords with EdgeListGraph using VanillaTextRank
    edge_list = EdgeListGraph()
    if encoded_tokens:
        edge_list.fill_from_tokens(encoded_tokens, 5)
    text_rank_edge = VanillaTextRank(edge_list)
    text_rank_edge.train()
    top_keywords_edge = text_rank_edge.get_top_keywords(10)
    print(encoded_top_edge := encoder.decode(top_keywords_edge))

    #   mark 8: extract keywords with AdjacencyMatrixGraph using PositionBiasedTextRank
    # if encoded_tokens:
    #     adjacency_graph.fill_positions(encoded_tokens)
    # adjacency_graph.calculate_position_weights()
    # bias_text_rank_adj = PositionBiasedTextRank(adjacency_graph)
    # bias_text_rank_adj.train()
    # top_keywords_adj_bias = bias_text_rank_adj.get_top_keywords(10)
    # print(encoded_top_adj_pos_bias := encoder.decode(top_keywords_adj_bias))

    #   mark 8: extract keywords with EdgeListGraph using PositionBiasedTextRank
    if encoded_tokens:
        edge_list.fill_positions(encoded_tokens)
    edge_list.calculate_position_weights()
    bias_text_rank_edge = PositionBiasedTextRank(edge_list)
    bias_text_rank_edge.train()
    top_keywords_edge_bias = bias_text_rank_edge.get_top_keywords(10)
    print(encoded_top_edge_pos_bias := encoder.decode(top_keywords_edge_bias))

    #   mark 10: comparing methods in csv file
    benchmark = KeywordExtractionBenchmark(eng_stop_words,
                                           tuple(punctuation),
                                           idf,
                                           BENCHMARK_MATERIALS_PATH)
    benchmark.run()
    benchmark.save_to_csv(PROJECT_ROOT / 'report.csv')

    RESULT = benchmark.report
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
