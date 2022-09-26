"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n, calculate_tf, calculate_tfidf


if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'


    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / 'Дюймовочка.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        target_text = file.read()
        clean_and_tokenize(target_text)
        tokens = clean_and_tokenize(target_text)
    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = file.read().split('\n')
        print(remove_stop_words(tokens, stop_words))
        calculate_frequencies(tokens)
        frequencies = calculate_frequencies(tokens)

    # reading IDF scores for all tokens in the corpus of H.C. Andersen tales
    IDF_PATH = ASSETS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)
        calculate_tf(frequencies)

    # reading frequencies for all tokens in the corpus of H.C. Andersen tales
    CORPUS_FREQ_PATH = ASSETS_PATH / 'corpus_frequencies.json'
    with open(CORPUS_FREQ_PATH, 'r', encoding='utf-8') as file:
        corpus_freqs = json.load(file)
        term_freq = calculate_tf(frequencies)
        calculate_tfidf(term_freq, idf)
        print("Топ 10 популярных слов в этом тексте: ", get_top_n(frequencies, 10))
    RESULT = None
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'


