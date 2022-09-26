"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
import main


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

    tokens = main.clean_and_tokenize(target_text)
    if tokens:
        tokens_with_no_sw = main.remove_stop_words(tokens, stop_words)
    if tokens_with_no_sw:
        frequencies = main.calculate_frequencies(tokens_with_no_sw)
    if frequencies:
        most_popular_words = main.get_top_n(frequencies, 16)
    print(most_popular_words)
    if frequencies:
        term_freq = main.calculate_tf(frequencies)
    if term_freq:
        tfidf_dict = main.calculate_tfidf(term_freq, idf)
    if tfidf_dict:
        top10_tfidf_dict = main.get_top_n(tfidf_dict, 10)
    print(top10_tfidf_dict)
    if frequencies:
        expected = main.calculate_expected_frequency(frequencies, corpus_freqs)
    if expected:
        chi_values_dict = main.calculate_chi_values(expected, frequencies)
    if chi_values_dict:
        significant_words = main.extract_significant_words(chi_values_dict, 0.01)
    if significant_words:
        top10_chi_values_dict = main.get_top_n(significant_words, 10)
    print(top10_chi_values_dict)

    RESULT = top10_chi_values_dict
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
