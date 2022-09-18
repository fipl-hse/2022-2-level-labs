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

    tokenized_text = main.clean_and_tokenize(target_text)

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = file.read().split('\n')

    tokenized_text = main.remove_stop_words(tokenized_text, stop_words)

    frequencies = main.calculate_frequencies(tokenized_text)

    tf_calculated = main.calculate_tf(frequencies)

    # reading IDF scores for all tokens in the corpus of H.C. Andersen tales
    IDF_PATH = ASSETS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    tfidf_calculated = main.calculate_tfidf(tf_calculated, idf)
    print(main.get_top_n(tfidf_calculated, 10))

    # reading frequencies for all tokens in the corpus of H.C. Andersen tales
    CORPUS_FREQ_PATH = ASSETS_PATH / 'corpus_frequencies.json'
    with open(CORPUS_FREQ_PATH, 'r', encoding='utf-8') as file:
        corpus_freqs = json.load(file)

    expected_frequency = main.calculate_expected_frequency(frequencies, corpus_freqs)
    chi_values = main.calculate_chi_values(expected_frequency, frequencies)
    significant_words = main.extract_significant_words(chi_values, 0.05)
    print(main.get_top_n(significant_words, 10))

    RESULT = main.get_top_n(significant_words, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
