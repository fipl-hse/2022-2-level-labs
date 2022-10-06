"""
Frequency-driven keyword extraction starter
"""
from pathlib import Path
import json
from lab_1_keywords_tfidf.main import (clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n,
                  calculate_tf, calculate_tfidf, calculate_expected_frequency)

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

    def str_to_float(str_var: dict[str, int]) -> dict[str, float]:
        dict_result = {}
        keys = list(str_var.keys())
        for key in keys:
            dict_result[key] = float(str_var[key])
        return dict_result


    text_step1 = clean_and_tokenize(target_text)
    if isinstance(text_step1, list):
        text_step2 = remove_stop_words(text_step1, stop_words)
        if isinstance(text_step2, list):
            text_step3 = calculate_frequencies(text_step2)
            if text_step3 is not None:
                text_step4 = get_top_n(str_to_float(text_step3), 10)
                text_step5 = calculate_tf(text_step3)
                text_step8 = calculate_expected_frequency(text_step3, corpus_freqs)
                if isinstance(text_step5, dict):
                    text_step6 = calculate_tfidf(text_step5, idf)
    #step 7
                    if text_step6 is not None:
                        print(get_top_n(text_step6, 10))


                        RESULT = get_top_n(text_step6, 10)
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'

