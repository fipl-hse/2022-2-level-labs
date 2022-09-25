"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path


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

    from lab_1_keywords_tfidf.main import (clean_and_tokenize, remove_stop_words, calculate_frequencies,
                                            get_top_n, calculate_tf, calculate_tfidf,
                                           calculate_expected_frequency, calculate_chi_values,
                                           extract_significant_words)

    TEXT_WITHOUT_PUNCTUATION = None
    TEXT_WITHOUT_STOP_WORDS = None
    WORDS_WITH_FREQUENCIES = None
    TF = None
    TFIDF = None
    RESULT = None
    WORDS_WITH_EXP_FREQ = None
    WORDS_WITH_CHI_VALUES = None
    SIGNIFICANT = None

    if target_text:
        TEXT_WITHOUT_PUNCTUATION = clean_and_tokenize(target_text)

    if TEXT_WITHOUT_PUNCTUATION:
        TEXT_WITHOUT_STOP_WORDS = remove_stop_words(TEXT_WITHOUT_PUNCTUATION, stop_words)

    if TEXT_WITHOUT_STOP_WORDS:
        WORDS_WITH_FREQUENCIES = calculate_frequencies(TEXT_WITHOUT_STOP_WORDS)

    if WORDS_WITH_FREQUENCIES:
        TF = calculate_tf(WORDS_WITH_FREQUENCIES)

    if TF and idf:
        TFIDF = calculate_tfidf(TF, idf)

    if TFIDF:
        top_tfidf_words = get_top_n(TFIDF, 10)

    if WORDS_WITH_FREQUENCIES and corpus_freqs:
        WORDS_WITH_EXP_FREQ = calculate_expected_frequency(WORDS_WITH_FREQUENCIES, corpus_freqs)

    if WORDS_WITH_EXP_FREQ and WORDS_WITH_FREQUENCIES:
        WORDS_WITH_CHI_VALUES = calculate_chi_values(WORDS_WITH_EXP_FREQ, WORDS_WITH_FREQUENCIES)

    if WORDS_WITH_CHI_VALUES:
        SIGNIFICANT = extract_significant_words(WORDS_WITH_CHI_VALUES, 0.05)

    if SIGNIFICANT:
        RESULT = get_top_n(SIGNIFICANT, 10)
        print(RESULT)

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
