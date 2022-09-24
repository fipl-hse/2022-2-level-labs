"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n,
    calculate_tf,
    calculate_tfidf,
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

    NO_STOP_WORDS = None
    FREQ_DICT = None
    TF_DICT = None
    TF_IDF = None
    CLEAN_AND_TOKENS = clean_and_tokenize(target_text)

    if CLEAN_AND_TOKENS:
        NO_STOP_WORDS = remove_stop_words(CLEAN_AND_TOKENS, stop_words)

    if NO_STOP_WORDS:
        FREQ_DICT = calculate_frequencies(NO_STOP_WORDS)
        print("Most frequent words in freq_dict:", get_top_n(FREQ_DICT, 8))

    if FREQ_DICT:
        TF_DICT = calculate_tf(FREQ_DICT)

    if TF_DICT:
        TF_IDF = calculate_tfidf(TF_DICT, idf)

    if TF_IDF:
        print("Most frequent words in tfidf_dict:", (get_top_n(TF_IDF, 10)))

    RESULT = TF_IDF
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
