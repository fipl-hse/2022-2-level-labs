"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words, calculate_frequencies, \
    get_top_n, calculate_tf, calculate_tfidf, calculate_expected_frequency, \
    calculate_chi_values, extract_significant_words


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

    CLEAN_TEXT, FREQUENCIES, TF_CALCULATED, TFIDF_CALCULATED, EXPECTED_FREQUENCY, \
        CHI_VALUES, SIGNIFICANT_WORDS, RESULT = [None for not_undefined in range(8)]

    TOKENIZED_TEXT = clean_and_tokenize(target_text)
    if TOKENIZED_TEXT:
        CLEAN_TEXT = remove_stop_words(TOKENIZED_TEXT, stop_words)
    if CLEAN_TEXT:
        FREQUENCIES = calculate_frequencies(CLEAN_TEXT)
    if FREQUENCIES:
        TF_CALCULATED = calculate_tf(FREQUENCIES)
    if TF_CALCULATED:
        TFIDF_CALCULATED = calculate_tfidf(TF_CALCULATED, idf)
    if TFIDF_CALCULATED:
        print(get_top_n(TFIDF_CALCULATED, 10))
    if FREQUENCIES:
        EXPECTED_FREQUENCY = calculate_expected_frequency(FREQUENCIES, corpus_freqs)
    if EXPECTED_FREQUENCY and FREQUENCIES:
        CHI_VALUES = calculate_chi_values(EXPECTED_FREQUENCY, FREQUENCIES)
    if CHI_VALUES:
        SIGNIFICANT_WORDS = extract_significant_words(CHI_VALUES, 0.05)
    if SIGNIFICANT_WORDS:
        print(get_top_n(SIGNIFICANT_WORDS, 10))
        RESULT = get_top_n(SIGNIFICANT_WORDS, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
