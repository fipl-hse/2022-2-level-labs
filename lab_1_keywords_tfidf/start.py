"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import (clean_and_tokenize, remove_stop_words, calculate_frequencies,
                                       get_top_n, calculate_tf, calculate_tfidf, calculate_expected_frequency,
                                       calculate_chi_values, extract_significant_words)


if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / 'Дюймовочка.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        target_text = file.read()

    tokens = clean_and_tokenize(target_text)
    print(tokens)

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = file.read().split('\n')

    res_tokens = remove_stop_words(tokens, stop_words)
    print(res_tokens)

    frequencies = calculate_frequencies(res_tokens)
    print(frequencies)

    top_10 = get_top_n(frequencies, 10)
    print(top_10)

    tf = calculate_tf(frequencies)
    print(tf)

    # reading IDF scores for all tokens in the corpus of H.C. Andersen tales
    IDF_PATH = ASSETS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    tfidf = calculate_tfidf(tf, idf)
    print(tfidf)

    top_10_tfidf = get_top_n(tfidf, 10)
    print(top_10_tfidf)

    # reading frequencies for all tokens in the corpus of H.C. Andersen tales
    CORPUS_FREQ_PATH = ASSETS_PATH / 'corpus_frequencies.json'
    with open(CORPUS_FREQ_PATH, 'r', encoding='utf-8') as file:
        corpus_freqs = json.load(file)

    expected = calculate_expected_frequency(frequencies, corpus_freqs)
    print(expected)

    chi_values = calculate_chi_values(expected, frequencies)
    print(chi_values)

    significant_words = extract_significant_words(chi_values, 0.05)
    print(significant_words)

    top_10_chi = get_top_n(significant_words, 10)
    print(top_10_chi)

    RESULT = get_top_n(significant_words, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
