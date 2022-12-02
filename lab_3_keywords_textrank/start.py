"""
TextRank keyword extraction starter
"""

from pathlib import Path
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

    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'
    TARGET_TEXT_PATH = ASSETS_PATH / 'article.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        text = file.read()
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

    # steps 6, 7.2, 9.3
    ADJ_GRAPH = AdjacencyMatrixGraph()
    EDJ_GRAPH = EdgeListGraph()
    for GRAPH in ADJ_GRAPH, EDJ_GRAPH:
        GRAPH.fill_from_tokens(ENCODED_TOKENS, 3)
        GRAPH.fill_positions(ENCODED_TOKENS)
        GRAPH.calculate_position_weights()

    for TEXTRANK in (VanillaTextRank(ADJ_GRAPH), VanillaTextRank(EDJ_GRAPH),
                     PositionBiasedTextRank(ADJ_GRAPH), PositionBiasedTextRank(EDJ_GRAPH)):
        print(TEXTRANK.__class__.__name__, end='. ')
        print(TEXTRANK.__getattribute__('_graph').__class__.__name__, end='. ')

        TEXTRANK.train()
        TOP_ENCODED_TOKENS = TEXTRANK.get_top_keywords(10)
        TOP_DECODED_TOKENS = ENCODER.decode(TOP_ENCODED_TOKENS)

        print(f'Top tokens: {TOP_DECODED_TOKENS}\n')

    # PositionBiasedTextRank is lower than VanillaTextRank. Both types extract different top tokens

    material_path = ASSETS_PATH / 'benchmark_materials'
    ENG_STOP_WORDS_PATH = ASSETS_PATH / 'benchmark_materials/eng_stop_words.txt'
    IDF_PATH = ASSETS_PATH / 'benchmark_materials/IDF.json'
    with (open(ENG_STOP_WORDS_PATH, 'r', encoding='utf-8') as stop_words_to_read,
          open(IDF_PATH, 'r', encoding='utf-8') as idf_to_read):
        eng_stop_words = tuple(stop_words_to_read.read().split('\n'))
        idf = json.load(idf_to_read)
    report = KeywordExtractionBenchmark(eng_stop_words, tuple('.,!?-:;()&'),
                                        idf, material_path)
    report_dict = report.run()
    REPORT_CSV = ASSETS_PATH / 'report.csv'
    report.save_to_csv(REPORT_CSV)

    RESULT = TOP_DECODED_TOKENS
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
