"""
TextRank keyword extraction starter
"""

import json
from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import TextPreprocessor, TextEncoder, AdjacencyMatrixGraph, VanillaTextRank, \
    EdgeListGraph, PositionBiasedTextRank, KeywordExtractionBenchmark

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

    PUNCTUATION_MARKS_LIST = []
    for punc in punctuation:
        PUNCTUATION_MARKS_LIST.append(punc)
    PUNCTUATION_MARKS = tuple(PUNCTUATION_MARKS_LIST)
    PREPROCESSED_TEXT = TextPreprocessor(stop_words, PUNCTUATION_MARKS)
    TOKENS = PREPROCESSED_TEXT.preprocess_text(text)
    TEXT_TO_CODE = TextEncoder()
    ENCODED_TXT = TEXT_TO_CODE.encode(TOKENS)

    GRAPH_OF_TEXT = AdjacencyMatrixGraph()
    EDGE_LIST_GRAPH = EdgeListGraph()
    for one_class_algoritm in GRAPH_OF_TEXT, EDGE_LIST_GRAPH:
        one_class_algoritm.fill_from_tokens(ENCODED_TXT, 5)
        one_class_algoritm.fill_positions(ENCODED_TXT)
        one_class_algoritm.calculate_position_weights()

    VANILLA_GRAPH_ADJA = VanillaTextRank(GRAPH_OF_TEXT)
    VANILLA_GRAPH_EDGE = VanillaTextRank(EDGE_LIST_GRAPH)
    POSITION_BIASED_ADJACENCY = PositionBiasedTextRank(GRAPH_OF_TEXT)
    POSITION_BIASED_EDGE = PositionBiasedTextRank(EDGE_LIST_GRAPH)

    LABELS = ['VanillaTextRank method for adjacency graph: ', 'VanillaTextRank method for edge graph: ',
              'Position biased method for adjacency graph: ', 'Position biased method for edge graph: ']
    IDX = 0

    for one_method in VANILLA_GRAPH_ADJA, VANILLA_GRAPH_EDGE, POSITION_BIASED_ADJACENCY, POSITION_BIASED_EDGE:
        one_method.train()
        TOP_WORDS = one_method.get_top_keywords(10)
        DECODED_WORDS = TEXT_TO_CODE.decode(TOP_WORDS)
        print(LABELS[IDX])
        print(DECODED_WORDS)
        IDX += 1

    IDF_PATH = ASSETS_PATH / 'benchmark_materials/IDF.json'
    with open(IDF_PATH, encoding='UTF-8') as FILE:
        JSON_FILE = json.load(FILE)
    ENG_PATH = ASSETS_PATH / 'benchmark_materials/eng_stop_words.txt'
    with open(ENG_PATH, encoding='UTF-8') as FILE:
        ENGLISH_STOPS = [line.rstrip('\n') for line in FILE]

    PATH_TO_BENCH = ASSETS_PATH / 'benchmark_materials'
    KEYWORD_EXTRACTION = KeywordExtractionBenchmark(tuple(ENGLISH_STOPS), PUNCTUATION_MARKS, JSON_FILE, PATH_TO_BENCH)
    KEYWORD_EXTRACTION.run()
    PATH_TO_CSV = 'lab_3_keywords_textrank/report.csv'
    KEYWORD_EXTRACTION.save_to_csv(PATH_TO_CSV)

    RESULT = DECODED_WORDS
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
