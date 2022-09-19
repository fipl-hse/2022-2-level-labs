"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path

from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n
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

    no_stop_words = None
    frequencies_dict = None
    get_top_six = None
    top = 6

    split_text = clean_and_tokenize(target_text)

    if split_text:
        no_stop_words = remove_stop_words(split_text, stop_words)

    if no_stop_words:
        frequencies_dict = calculate_frequencies(no_stop_words)

    if frequencies_dict:
        get_top_six = get_top_n(frequencies_dict, top)


    RESULT = get_top_six
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'