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

    do_clean_and_tokenized = clean_and_tokenize(target_text)
    if do_clean_and_tokenized:
        do_stop_words = remove_stop_words(do_clean_and_tokenized, stop_words)
    if do_stop_words:
        do_freqs = calculate_frequencies(do_stop_words)
    if do_freqs:
        do_tf = calculate_tf(do_freqs)
    if do_tf:
        do_tfidf = calculate_tfidf(do_tf, idf)
    if do_tfidf and do_freqs:
        print(get_top_n(do_tfidf, 10))
        do_expected_frequency = calculate_expected_frequency(do_freqs, corpus_freqs)
    if do_expected_frequency and do_freqs:
        do_chi_value = calculate_chi_values(do_expected_frequency, do_freqs)
    if do_chi_value:
        do_extract_significant_words = extract_significant_words(do_chi_value, 0.001)
    if do_extract_significant_words:
        print(get_top_n(do_extract_significant_words, 10))
        RESULT = get_top_n(do_extract_significant_words, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
