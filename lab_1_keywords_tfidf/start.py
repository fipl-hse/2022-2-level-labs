"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import (clean_and_tokenize,
                                       remove_stop_words,
                                       calculate_frequencies,
                                       get_top_n,
                                       calculate_tf,
                                       calculate_tfidf,
                                       calculate_expected_frequency,
                                       calculate_chi_values)

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

    RESULT = None
    TOKENS = \
        FREQUENCIES = \
        CALCULATED_TF = \
        CALCULATED_TFIDF = \
        EXPECTED_FREQUENCY = \
        TOP_CHI = \
        CHI_VALUES = None

    split_text = clean_and_tokenize(target_text)

    if split_text and stop_words:
        TOKENS = remove_stop_words(split_text, stop_words)

    if TOKENS:
        FREQUENCIES = calculate_frequencies(TOKENS)

    if FREQUENCIES:
        CALCULATED_TF = calculate_tf(FREQUENCIES)

    if CALCULATED_TF and idf:
        CALCULATED_TFIDF = calculate_tfidf(CALCULATED_TF, idf)

    if CALCULATED_TFIDF:
        print("The most frequent words by 'calculated_tfidf': ", get_top_n(CALCULATED_TFIDF, 10))

    if FREQUENCIES and corpus_freqs:
        EXPECTED_FREQUENCY = calculate_expected_frequency(FREQUENCIES, corpus_freqs)

    if EXPECTED_FREQUENCY and FREQUENCIES:
        CHI_VALUES = calculate_chi_values(EXPECTED_FREQUENCY, FREQUENCIES)

    if CHI_VALUES:
        TOP_CHI = get_top_n(CHI_VALUES, 10)

    if TOP_CHI:
        print("The most frequent words by 'chi values': ", TOP_CHI)

    RESULT = TOP_CHI
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
