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
    calculate_tfidf,
    calculate_expected_frequency,
    calculate_chi_values
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

    NOT_STOP_WORDS = None
    FREQ_DICT = None
    TF_DICT = None
    EXPECTED_DICT = None
    X2_DICT = None
    TOP_X2 = None

    split_words = clean_and_tokenize(target_text)

    if split_words:
        NOT_STOP_WORDS = remove_stop_words(split_words, stop_words)

    if NOT_STOP_WORDS:
        FREQ_DICT = calculate_frequencies(NOT_STOP_WORDS)

    if FREQ_DICT:
        top_words = get_top_n(FREQ_DICT, 10)
        TF_DICT = calculate_tf(FREQ_DICT)

    if TF_DICT:
        tfidf = calculate_tfidf(TF_DICT, idf)
        print(f'The top of words by tfidf: {get_top_n(tfidf, 10)}')

    if FREQ_DICT:
        EXPECTED_DICT = calculate_expected_frequency(FREQ_DICT, corpus_freqs)

    if EXPECTED_DICT and FREQ_DICT:
        X2_DICT = calculate_chi_values(EXPECTED_DICT, FREQ_DICT)

    if X2_DICT:
        TOP_X2 = get_top_n(X2_DICT, 10)
        print(f'The top of words by chi: {TOP_X2}')

    RESULT = TOP_X2
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
