"""
Frequency-driven keyword extraction starter
"""

import json
from pathlib import Path


from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n,
    calculate_tf,
    calculate_tfidf
)


if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / 'Дюймовочка.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = file.read().split('\n')

    # reading IDF scores for all tokens in the corpus of H.C. Andersen tales
    IDF_PATH = ASSETS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    # reading frequencies for all tokens in the corpus of H.C. Andersen tales
    CORPUS_FREQ_PATH = ASSETS_PATH / 'corpus_frequencies.json'
    with open(CORPUS_FREQ_PATH, 'r', encoding='utf-8') as file:
        corpus_freqs = json.load(file)


    NO_STOP_WORDS = None
    FREQUENCIES_DICT = None
    GET_TOP_TEN = None
    COUNT_TF = None
    COUNT_TDIDF = None
    TOP = 10


    SPLIT_TEXT = clean_and_tokenize(target_text)

    if SPLIT_TEXT:
        NO_STOP_WORDS = remove_stop_words(SPLIT_TEXT, stop_words)

    if NO_STOP_WORDS:
        FREQUENCIES_DICT = calculate_frequencies(NO_STOP_WORDS)

    if FREQUENCIES_DICT:
        GET_TOP_TEN = get_top_n(FREQUENCIES_DICT, TOP)

    if GET_TOP_TEN:
        COUNT_TF = calculate_tf(FREQUENCIES_DICT)

    if COUNT_TF:
        COUNT_TDIDF = calculate_tfidf(COUNT_TF, idf)

    if COUNT_TDIDF:
        GET_TOP_TEN = get_top_n(COUNT_TDIDF, TOP)

    RESULT = GET_TOP_TEN
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'

    print(RESULT)
