"""
TextRank keyword extraction starter
"""

from pathlib import Path
from lab_3_keywords_textrank.main import (
    TextPreprocessor,
    TextEncoder,
    extract_pairs,
    AdjacencyMatrixGraph,
    VanillaTextRank,
    EdgeListGraph,
    # PositionBiasedTextRank,
    # KeywordExtractionBenchmark
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
        print(extract_pairs(ENCODED_TOKENS, 3))

    ADJ_GRAPH = AdjacencyMatrixGraph()
    EDGE_GRAPH = EdgeListGraph()
    for GRAPH in ADJ_GRAPH, EDGE_GRAPH:
        GRAPH.fill_from_tokens(ENCODED_TOKENS, 3)
        TEXTRANK = VanillaTextRank(GRAPH)
        TEXTRANK.train()
        TOP_ENCODED_TOKENS = TEXTRANK.get_top_keywords(10)
        TOP_DECODED_TOKENS = ENCODER.decode(TOP_ENCODED_TOKENS)
        print(f'Top tokens: {TOP_DECODED_TOKENS}')

    RESULT = TOP_DECODED_TOKENS
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
