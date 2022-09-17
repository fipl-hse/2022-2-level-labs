"""
Frequency-driven keyword extraction starter
"""
from pathlib import Path
import json
import main


if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path('new').parent
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

    print(main.clean_and_tokenize(target_text))
    tokens = main.clean_and_tokenize(target_text)
    print(main.remove_stop_words(tokens, stop_words))
    tokens = main.remove_stop_words(tokens, stop_words)
    print(main.calculate_frequencies(tokens))
    frequencies = main.calculate_frequencies(tokens)
    print(main.get_top_n(frequencies, 8))
    print(main.calculate_tf(frequencies))
    term_freq = main.calculate_tf(frequencies)
    print(main.calculate_tfidf(term_freq, idf))
    tfidf_dic = main.calculate_tfidf(term_freq, idf)
    print(main.get_top_n(tfidf_dic, 10))
    print(main.calculate_expected_frequency(frequencies, corpus_freqs))
    expected_freq = main.calculate_expected_frequency(frequencies, corpus_freqs)
    print(main.calculate_chi_values(expected_freq, frequencies))


    # RESULT = None
    # # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    # assert RESULT, 'Keywords are not extracted'


