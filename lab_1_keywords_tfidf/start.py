"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import (clean_and_tokenize, remove_stop_words, calculate_frequencies,
                                            get_top_n, calculate_tf, calculate_tfidf,
                                           calculate_expected_frequency, calculate_chi_values,
                                           extract_significant_words)


if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / "assets"

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / "Дюймовочка.txt"
    with open(TARGET_TEXT_PATH, "r", encoding="utf-8") as file:
        target_text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / "stop_words.txt"
    with open(STOP_WORDS_PATH, "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")

    # reading IDF scores for all tokens in the corpus of H.C. Andersen tales
    IDF_PATH = ASSETS_PATH / "IDF.json"
    with open(IDF_PATH, "r", encoding="utf-8") as file:
        idf = json.load(file)

    # reading frequencies for all tokens in the corpus of H.C. Andersen tales
    CORPUS_FREQ_PATH = ASSETS_PATH / "corpus_frequencies.json"
    with open(CORPUS_FREQ_PATH, "r", encoding="utf-8") as file:
        corpus_freqs = json.load(file)

    no_stop_words_text, frequency, tf, tfidf, expected_frequency, chi_values, significant_words = \
        [None for undef in range(7)]

    cleaned_text = clean_and_tokenize(target_text)

    if cleaned_text:
        no_stop_words_text = remove_stop_words(cleaned_text, stop_words)

    if no_stop_words_text:
        frequency = calculate_frequencies(no_stop_words_text)

    if frequency:
        tf = calculate_tf(frequency)
        expected_frequency = calculate_expected_frequency(frequency, corpus_freqs)
        top_of_words = get_top_n(frequency, 10.0)

    if tf:
        tfidf = calculate_tfidf(tf, idf)

    if tfidf:
        tfidf_top = get_top_n(tfidf, 10)
        print(tfidf_top)

    if expected_frequency and frequency:
        chi_values = calculate_chi_values(expected_frequency, frequency)

    if chi_values:
        significant_words = extract_significant_words(chi_values, 0.001)

    if significant_words:
        RESULT = get_top_n(significant_words, 10)
        # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
        assert RESULT, 'Keywords are not extracted'
