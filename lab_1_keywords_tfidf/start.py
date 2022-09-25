"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n,
    calculate_tf,
    calculate_tfidf,
    calculate_expected_frequency,
    calculate_chi_values
)

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

    no_stop_words, freq_dict, tf_dict, tfidf_dict, exp_freq_dict, chi_dict = [None for notdef in range(6)]
    tokenization = clean_and_tokenize(target_text)

    if tokenization:
        no_stop_words = remove_stop_words(tokenization, stop_words)

    if no_stop_words:
        freq_dict = calculate_frequencies(no_stop_words)

    if freq_dict:
        tf_dict = calculate_tf(freq_dict)

    if freq_dict and tf_dict:
        tfidf_dict = calculate_tfidf(tf_dict, idf)

    if tfidf_dict and freq_dict:
        exp_freq_dict = calculate_expected_frequency(freq_dict, corpus_freqs)

    if exp_freq_dict:
        chi_dict = calculate_chi_values(exp_freq_dict, freq_dict)

    if chi_dict:
        RESULT = get_top_n(chi_dict, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
