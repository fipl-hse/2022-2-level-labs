"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import (clean_and_tokenize, remove_stop_words, calculate_frequencies,
                                       calculate_tf, calculate_tfidf, calculate_expected_frequency,
                                       calculate_chi_values, extract_significant_words, get_top_n)

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

    (tokens, frequencies, top_n_freq, term_freq, tfidf_dict,
     top_n_tfidf, expected, chi_values, significant_words_1, top_n_chi) = [None for variable in range(10)]

    clean_tokens = clean_and_tokenize(target_text)
    if clean_tokens:
        tokens = remove_stop_words(clean_tokens, stop_words)
    if tokens:
        frequencies = calculate_frequencies(tokens)
    if frequencies:
        # по заданию требуется продемонстрировать топ слов не просто согласно
        # частотам, а согласно скорам tf-idf, которые являются  значениями с
        # плавающей точкой. В случае передачи целочисленных значений получается
        # вот такая ошибка. Это не очень интуитивно исходя из прописанных тайпингов,
        # но если делать строго по заданию, то проблем быть не должно
        top_n_freq = get_top_n(frequencies, 10)
    if frequencies:
        term_freq = calculate_tf(frequencies)
    if term_freq:
        tfidf_dict = calculate_tfidf(term_freq, idf)
    if tfidf_dict:
        top_n_tfidf = get_top_n(tfidf_dict, 10)
        print(top_n_tfidf)
    if frequencies:
        expected = calculate_expected_frequency(frequencies, corpus_freqs)
    if expected and frequencies:
        chi_values = calculate_chi_values(expected, frequencies)
    if chi_values:
        significant_words_1 = extract_significant_words(chi_values, 0.001)
    if significant_words_1:
        top_n_chi = get_top_n(significant_words_1, 10)
        print(top_n_chi)

    RESULT = top_n_chi
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
