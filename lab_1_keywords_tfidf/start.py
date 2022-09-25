"""
Frequency-driven keyword extraction starter
"""
import main
from pathlib import Path
import json

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

    RESULT = None

    text_step1 = main.clean_and_tokenize(target_text)
    text_step2 = main.remove_stop_words(text_step1, stop_words)
    text_step3 = main.calculate_frequencies(text_step2)
    text_step4 = main.get_top_n(text_step3, 10)
    text_step5 = main.calculate_tf(text_step3)
    text_step6 = main.calculate_tfidf(text_step5, idf)
    RESULT = text_step6
    print(text_step1)
    print(text_step2)
    print(text_step3)
    print(text_step4)
    print(text_step5)
    print(text_step6)

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'

