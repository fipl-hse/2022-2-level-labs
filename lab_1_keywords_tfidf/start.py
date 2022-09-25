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
    print(tokens)
    cleaned_tokens = main.remove_stop_words(tokens, stop_words)
    print(cleaned_tokens)
    frequencies = main.calculate_frequencies(cleaned_tokens)
    print(frequencies)
    top_list = main.get_top_n(frequencies, 10)
    print(top_list)
    term_frequencies = main.calculate_tf(frequencies)
    print(term_frequencies)
    tdidf = main.calculate_tfidf(term_frequencies, idf)
    print(tdidf)
    print(main.get_top_n(tdidf, 10))
    expected_frequencies = main.calculate_expected_frequency(frequencies, corpus_freqs)
    print(expected_frequencies)
    chi_values = main.calculate_chi_values(expected_frequencies, frequencies)
    print(chi_values)
    significant = main.extract_significant_words(chi_values, 0.05)
    top_chi = main.get_top_n(chi_values, 10)
    print(top_chi)


    RESULT = top_chi
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'


