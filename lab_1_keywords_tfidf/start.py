"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path


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

    from lab_1_keywords_tfidf.main import (clean_and_tokenize, remove_stop_words, calculate_frequencies,
                                            calculate_expected_frequency, extract_significant_words,
                                            get_top_n, calculate_chi_values, calculate_tf, calculate_tfidf)

    text_without_punctuation = None
    text_without_stop_words = None
    words_with_frequencies = None
    tf = None
    tfidf = None
    RESULT = None

    if target_text:
        text_without_punctuation = clean_and_tokenize(target_text)

    if text_without_punctuation:
        text_without_stop_words = remove_stop_words(text_without_punctuation, stop_words)

    if text_without_stop_words:
        words_with_frequencies = calculate_frequencies(text_without_stop_words)

    if words_with_frequencies:
        tf = calculate_tf(words_with_frequencies)

    if tf and idf:
        tfidf = calculate_tfidf(tf, idf)

    if tfidf:
        RESULT = get_top_n(tfidf, 10)
        print(RESULT)

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
