"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import (clean_and_tokenize, remove_stop_words,
                                       calculate_frequencies, get_top_n, calculate_tf, calculate_tfidf,
                                       calculate_expected_frequency, calculate_chi_values)

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

    FREQ_DICT, TF_DICT, TFIDF_DICT, TOP, EXPECTED_FREQ_DICT, CHI_DICT, TOP_CHI = [None for _ in range(7)]
    TOKENS = clean_and_tokenize(target_text)

    if TOKENS:
        TOKENS = remove_stop_words(TOKENS, stop_words)

    if TOKENS:
        FREQ_DICT = calculate_frequencies(TOKENS)

    if FREQ_DICT:
        TF_DICT = calculate_tf(FREQ_DICT)

    if TF_DICT:
        TFIDF_DICT = calculate_tfidf(TF_DICT, idf)

    if TFIDF_DICT:
        TOP = get_top_n(TFIDF_DICT, 10)

    if TOP:
        print('Most frequent words by tfidf_dict:', ', '.join(TOP), end='.\n')

    if FREQ_DICT:
        EXPECTED_FREQ_DICT = calculate_expected_frequency(FREQ_DICT, corpus_freqs)

    if EXPECTED_FREQ_DICT and FREQ_DICT:
        CHI_DICT = calculate_chi_values(EXPECTED_FREQ_DICT, FREQ_DICT)

    if CHI_DICT:
        TOP_CHI = get_top_n(CHI_DICT, 10)

    if TOP_CHI:
        print('Most frequent words by chi value:', ', '.join(TOP_CHI), end='.\n')

    RESULT = TOP_CHI
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
