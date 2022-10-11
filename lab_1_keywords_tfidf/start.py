"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import (clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n,
                  calculate_tf, calculate_tfidf, calculate_expected_frequency,
                  calculate_chi_values, extract_significant_words)


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

    CLEAN_TOKENS = None
    FREQUENCIES = None
    TF_DICT = None
    TFIDF_DICT = None
    EXP_FREQ_DICT = None
    CHI_DICT = None
    SIGNIFICANT_WORDS = None
    RESULT = None

    clean_text = clean_and_tokenize(target_text)

    if clean_text:
        CLEAN_TOKENS = remove_stop_words(clean_text, stop_words)

    if CLEAN_TOKENS:
        FREQUENCIES = calculate_frequencies(CLEAN_TOKENS)

    if FREQUENCIES:
        TF_DICT = calculate_tf(FREQUENCIES)

    if TF_DICT:
        TFIDF_DICT = calculate_tfidf(TF_DICT, idf)

    if TFIDF_DICT:
        top_10_tfidf = get_top_n(TFIDF_DICT, 10)

    if FREQUENCIES:
        EXP_FREQ_DICT = calculate_expected_frequency(FREQUENCIES, corpus_freqs)

    if EXP_FREQ_DICT and FREQUENCIES:
        CHI_DICT = calculate_chi_values(EXP_FREQ_DICT, FREQUENCIES)

    if CHI_DICT:
        SIGNIFICANT_WORDS = extract_significant_words(CHI_DICT, 0.05)

    RESULT = get_top_n(SIGNIFICANT_WORDS, 10) if SIGNIFICANT_WORDS else None
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
