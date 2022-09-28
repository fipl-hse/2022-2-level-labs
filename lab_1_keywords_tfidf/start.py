"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from main import clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n, calculate_tf, calculate_tfidf

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

no_stop_words_list = None
top_freqs_list = None
tf_freqs_dict = None
tfidf_freqs_dict = None

token_list = clean_and_tokenize(target_text)

if token_list:
    no_stop_words_list = remove_stop_words(token_list, stop_words)
if no_stop_words_list:
    top_freqs_list = calculate_frequencies(no_stop_words_list)
if top_freqs_list:
    tf_freqs_dict = calculate_tf(top_freqs_list)
if tf_freqs_dict:
    tfidf_freqs_dict = calculate_tfidf(tf_freqs_dict, idf)
if tfidf_freqs_dict:
    print(get_top_n(tfidf_freqs_dict, 10))

    RESULT = get_top_n(tfidf_freqs_dict, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
