"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import clean_and_tokenize, \
    remove_stop_words, calculate_frequencies, \
    get_top_n, calculate_tf, calculate_tfidf, \
    calculate_expected_frequency, calculate_chi_values, extract_significant_words


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

    ALPHA = 0.001
    TOKENIZED = clean_and_tokenize(target_text)
    if TOKENIZED:
        CLEANED = remove_stop_words(TOKENIZED, stop_words)
    if CLEANED:
        FREQUENCY = calculate_frequencies(CLEANED)
    if FREQUENCY:
        CALCULATED_TF = calculate_tf(FREQUENCY)
    if CALCULATED_TF:
        CALCULATED_TFIDF = calculate_tfidf(CALCULATED_TF, idf)
    if CALCULATED_TFIDF and FREQUENCY:
        print(get_top_n(CALCULATED_TFIDF, 10))
        EXPECTED_FREQUENCY = calculate_expected_frequency(FREQUENCY, corpus_freqs)
    if EXPECTED_FREQUENCY and FREQUENCY:
        CHI_VALUE = calculate_chi_values(EXPECTED_FREQUENCY, FREQUENCY)
    if CHI_VALUE:
        SIGNIFIC_WORDS = extract_significant_words(CHI_VALUE, ALPHA)
    if SIGNIFIC_WORDS:
        print(get_top_n(SIGNIFIC_WORDS, 10))
        RESULT = get_top_n(SIGNIFIC_WORDS, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
