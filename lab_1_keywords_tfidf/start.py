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

    tokenized_text, frequencies, tf_calculated, tfidf_calculated, expected_frequency, \
        chi_values, significant_words, RESULT = [None for not_undefined in range(8)]

    if target_text:
        tokenized_text = clean_and_tokenize(target_text)
    if tokenized_text and stop_words:
        tokenized_text = remove_stop_words(tokenized_text, stop_words)
    if tokenized_text:
        frequencies = calculate_frequencies(tokenized_text)
    if frequencies:
        tf_calculated = calculate_tf(frequencies)
    if tf_calculated and idf:
        tfidf_calculated = calculate_tfidf(tf_calculated, idf)
    if tfidf_calculated:
        print(get_top_n(tfidf_calculated, 10))
    if frequencies and corpus_freqs:
        expected_frequency = calculate_expected_frequency(frequencies, corpus_freqs)
    if expected_frequency and frequencies:
        chi_values = calculate_chi_values(expected_frequency, frequencies)
    if chi_values:
        significant_words = extract_significant_words(chi_values, 0.05)
    if significant_words:
        print(get_top_n(significant_words, 10))
        RESULT = get_top_n(significant_words, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
