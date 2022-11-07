"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from main import TextPreprocessor, TextEncoder, extract_pairs, AdjacencyMatrixGraph, VanillaTextRank

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
    GRAPH_OF_TEXT = AdjacencyMatrixGraph()
    for one_pair in TEXT_PAIRS:
        GRAPH_OF_TEXT.add_edge(one_pair[0], one_pair[1])
    VANILLA_GRAPH = VanillaTextRank(GRAPH_OF_TEXT)
    VANILLA_GRAPH.train()
    BEST_TOKENS = VANILLA_GRAPH.get_top_keywords(10)
    DECODED_TOKENS = TEXT_TO_CODE.decode(BEST_TOKENS)
    RESULT = DECODED_TOKENS
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'

