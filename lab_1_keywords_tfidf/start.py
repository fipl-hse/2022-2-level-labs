"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from main import (clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n,
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

    clean_text = clean_and_tokenize(target_text)
    print(clean_text)

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = file.read().split('\n')

    clean_tokens = remove_stop_words(clean_text,stop_words)
    print(clean_tokens)
    frequencies = calculate_frequencies(clean_tokens)
    print(frequencies)
    top_10 = get_top_n(frequencies, 10)
    print(top_10)
    tf_dict = calculate_tf(frequencies)
    print(tf_dict)

    # reading IDF scores for all tokens in the corpus of H.C. Andersen tales
    IDF_PATH = ASSETS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    tfidf_dict = calculate_tfidf(tf_dict, idf)
    print(tfidf_dict)

    top_10_tfidf = get_top_n(tfidf_dict, 10)
    print(top_10_tfidf)

    # reading frequencies for all tokens in the corpus of H.C. Andersen tales
    CORPUS_FREQ_PATH = ASSETS_PATH / 'corpus_frequencies.json'
    with open(CORPUS_FREQ_PATH, 'r', encoding='utf-8') as file:
        corpus_freqs = json.load(file)

    exp_freq_dict = calculate_expected_frequency(frequencies, corpus_freqs)
    print(exp_freq_dict)

    chi_dict = calculate_chi_values(exp_freq_dict, frequencies)
    print(chi_dict)

    significant_words = extract_significant_words(chi_dict, 0.05)
    print(significant_words)

    top_10_chi = get_top_n(significant_words, 10)
    print(top_10_chi)

    RESULT = None
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
