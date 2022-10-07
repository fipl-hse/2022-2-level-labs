"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path

from lab_1_keywords_tfidf import main

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

    if target_text:
        tok = main.clean_and_tokenize(target_text)
        print(tok)

    if tok:
        st_wo = main.remove_stop_words(tok, stop_words)
        print(st_wo)

    if st_wo:
        freq_dict = main.calculate_frequencies(st_wo)
        print(freq_dict)

    if freq_dict:
        tf_dict = main.calculate_tf(freq_dict)
        print(tf_dict)

    if tf_dict:
        tfidf_dict = main.calculate_tfidf(tf_dict, idf)
        print(tfidf_dict)

    if tfidf_dict:
        top_top = main.get_top_n(tfidf_dict, 10)
        print(top_top)

    RESULT = top_top
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
