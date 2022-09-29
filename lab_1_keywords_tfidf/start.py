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

    if target_text:
        tokens = clean_and_tokenize(target_text)
        print(tokens)
    if tokens:
        cleaned_tokens = remove_stop_words(tokens, stop_words)
        print(cleaned_tokens)
    if cleaned_tokens:
        frequencies = calculate_frequencies(cleaned_tokens)
        print(frequencies)
    if frequencies:
        top_list = get_top_n(frequencies, 10)
        print(top_list)
        term_frequencies = calculate_tf(frequencies)
        print(term_frequencies)
    if term_frequencies:
        tdidf = calculate_tfidf(term_frequencies, idf)
        print(tdidf)
    if tdidf:
        print(get_top_n(tdidf, 10))
    if frequencies:
        expected_frequencies = calculate_expected_frequency(frequencies, corpus_freqs)
        print(expected_frequencies)
    if expected_frequencies:
        chi_values = calculate_chi_values(expected_frequencies, frequencies)
        print(chi_values)
    if chi_values:
        significant = extract_significant_words(chi_values, 0.05)
        print(significant)
        top_chi = get_top_n(chi_values, 10)
        print(top_chi)

    RESULT = top_chi
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
