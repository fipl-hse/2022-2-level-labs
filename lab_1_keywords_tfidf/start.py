"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
# import main
from lab_1_keywords_tfidf.main import (clean_and_tokenize,
                                       remove_stop_words,
                                       calculate_frequencies,
                                       get_top_n,
                                       calculate_tf,
                                       calculate_tfidf)

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

    # tokens = main.clean_and_tokenize(target_text)
    # tokens = main.remove_stop_words(tokens, stop_words)
    # dictionary = main.calculate_frequencies(tokens)
    # print(main.get_top_n(dictionary, 6))
    # term_freq = main.calculate_tf(dictionary)
    # tf_idf_dict = main.calculate_tfidf(term_freq, idf)
    # print(main.get_top_n(tf_idf_dict, 10))


    WITHOUT_STOPWORDS = None
    FREQUENCY_DICTIONARY = None
    TOP_N_WORDS = 10
    TF_DICTIONARY = None
    TF_IDF = None
    CLEAN_TEXT = clean_and_tokenize(target_text)
    EXPECT_DICT = None

    if CLEAN_TEXT:
        WITHOUT_STOPWORDS = remove_stop_words(CLEAN_TEXT, stop_words)

    if WITHOUT_STOPWORDS:
        FREQUENCY_DICTIONARY = calculate_frequencies(WITHOUT_STOPWORDS)
        print(get_top_n(FREQUENCY_DICTIONARY, TOP_N_WORDS))

    if FREQUENCY_DICTIONARY:
        TF_DICTIONARY = calculate_tf(FREQUENCY_DICTIONARY)

    if TF_DICTIONARY:
        TF_IDF = calculate_tfidf(TF_DICTIONARY, idf)

    if TF_IDF:
        print(get_top_n(TF_IDF, 10))

    if EXPECT_DICT:
        EXPECT_DICT = calculate_expected_frequency(EXPECT_DICT, corpus_freqs)

    RESULT = EXPECT_DICT
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
