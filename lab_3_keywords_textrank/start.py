"""
TextRank keyword extraction starter
"""

from pathlib import Path
from time import process_time

import json

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

    PREPROCESSOR = TextPreprocessor(stop_words, tuple('.,!?-:;()'))
    TOKENS = PREPROCESSOR.preprocess_text(text)

    ENCODER = TextEncoder()
    ENCODED_TOKENS = ENCODER.encode(TOKENS)

    # step 3
    if ENCODED_TOKENS:
        print(f'Extracted pairs: {extract_pairs(ENCODED_TOKENS, 3)}\n')

    # # steps 6, 7.2, 9.3
    for GRAPH in AdjacencyMatrixGraph(), EdgeListGraph():
        GRAPH.fill_from_tokens(ENCODED_TOKENS, 3)
        GRAPH.fill_positions(ENCODED_TOKENS)
        GRAPH.calculate_position_weights()
        print(f'The graph is {GRAPH.__class__.__name__}.', end=' ')

        for TEXTRANK in (VanillaTextRank(GRAPH), PositionBiasedTextRank(GRAPH)):
            print(f'The textrank algorithm is {TEXTRANK.__class__.__name__}.', end=' ')
            time_start = process_time()
            TEXTRANK.train()
            TOP_ENCODED_TOKENS = TEXTRANK.get_top_keywords(10)
            TOP_DECODED_TOKENS = ENCODER.decode(TOP_ENCODED_TOKENS)
            time_stop = process_time()
            print(f'Elapsed in {time_stop - time_start} seconds.')
            print(f'Top tokens: {TOP_DECODED_TOKENS}\n')

    # PositionBiasedTextRank is lower than VanillaTextRank. Both types extract different top tokens

    MATERIALS_PATH = ASSETS_PATH / 'benchmark_materials'
    ENG_STOP_WORDS_PATH = MATERIALS_PATH / 'eng_stop_words.txt'
    IDF_PATH = MATERIALS_PATH / 'IDF.json'
    with (open(ENG_STOP_WORDS_PATH, 'r', encoding='utf-8') as stop_words_to_read,
          open(IDF_PATH, 'r', encoding='utf-8') as idf_to_read):
        eng_stop_words = tuple(stop_words_to_read.read().split('\n'))
        idf = json.load(idf_to_read)

    BENCHMARK = KeywordExtractionBenchmark(eng_stop_words, tuple('.,!?-:;()&'), idf, MATERIALS_PATH)
    BENCHMARK.run()
    BENCHMARK.save_to_csv(MATERIALS_PATH)

    RESULT = TOP_DECODED_TOKENS
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
