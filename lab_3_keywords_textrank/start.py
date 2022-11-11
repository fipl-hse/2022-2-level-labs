"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from main import TextPreprocessor, TextEncoder, extract_pairs, AdjacencyMatrixGraph, VanillaTextRank, EdgeListGraph

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
    TEXT_PAIRS = extract_pairs(ENCODED_TXT, 8)

    # step 6

    GRAPH_OF_TEXT = AdjacencyMatrixGraph()
    GRAPH_OF_TEXT.fill_from_tokens(ENCODED_TXT, 5)
    VANILLA_GRAPH_ADJA = VanillaTextRank(GRAPH_OF_TEXT)
    VANILLA_GRAPH_ADJA.train()


    # step 7.3

    EDGE_LIST_GRAPH = EdgeListGraph()
    EDGE_LIST_GRAPH.fill_from_tokens(ENCODED_TXT, 5)
    VANILLA_GRAPH_EDGE = VanillaTextRank(EDGE_LIST_GRAPH)
    VANILLA_GRAPH_EDGE.train()

    for i in set(ENCODED_TXT):
        if not EDGE_LIST_GRAPH.calculate_inout_score(i) == GRAPH_OF_TEXT.calculate_inout_score(i):
            print(i)


    BEST_TOKENS_ADJA = VANILLA_GRAPH_ADJA.get_top_keywords(10)
    DECODED_TOKENS_ADJA = TEXT_TO_CODE.decode(BEST_TOKENS_ADJA)
    BEST_TOKENS_EDGE = VANILLA_GRAPH_EDGE.get_top_keywords(10)
    DECODED_TOKENS_EDGE = TEXT_TO_CODE.decode(BEST_TOKENS_EDGE)

    print(DECODED_TOKENS_ADJA)
    print(DECODED_TOKENS_EDGE)

    RESULT = DECODED_TOKENS_ADJA
    # print(RESULT)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'

