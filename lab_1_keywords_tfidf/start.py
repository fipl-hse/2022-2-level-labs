"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path

from main import clean_and_tokenize, \
    remove_stop_words, calculate_frequencies,\
    get_top_n, calculate_tf, calculate_tfidf, \
    calculate_expected_frequency, calculate_chi_values, \
    extract_significant_words

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

    words_list = clean_and_tokenize(target_text)
    print("cписок слов: {} ".format(words_list))

    tokens = remove_stop_words(words_list, stop_words)
    print("слова без стоп-слов: {}".format(tokens))

    frequencies = calculate_frequencies(tokens)
    print("Частоты слов: {}".format(frequencies))

    print(get_top_n(frequencies, 8))

    term_freq = calculate_tf(frequencies)
    print("term frequency: ", term_freq)

    tf_idf = calculate_tfidf(term_freq, idf)
    print("tf_idf: ", tf_idf)

    print(get_top_n(tf_idf, 10))

    expected = (calculate_expected_frequency(frequencies, corpus_freqs))
    print("expected frequency: ", expected)

    chi_values = (calculate_chi_values(expected, frequencies))
    print("chi values: ", chi_values)

    print(extract_significant_words(chi_values, 0.05))

    print(get_top_n(chi_values, 10))
    RESULT = get_top_n(chi_values, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
