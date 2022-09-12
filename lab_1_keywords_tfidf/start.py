"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path

from typing import Optional, Union


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


    def clean_and_tokenize(text: str) -> Optional[list[str]]:
        if isinstance(text, str):
            no_punc_text = ''
            punctuation = '!?-.,\'\"():;'
            for e in text:
                if e not in punctuation:
                    no_punc_text += e
            all_words = no_punc_text.lower().strip().split()
            return all_words
        else:
            return None


    all_words = clean_and_tokenize(target_text)


    def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
        if isinstance(tokens, list) and isinstance(stop_words, list):
            no_stop_words = [w for w in tokens if w not in stop_words]
            print(no_stop_words)
            return no_stop_words
        else:
            return None


    no_stop_words = remove_stop_words(all_words, stop_words)


    def calculate_frequencies(tokens: list[str]) -> dict[str: int]:
        if (isinstance(tokens, list) and tokens
                and all(isinstance(w, str) for w in tokens)):
            freq_dict = {}
            for w in tokens:
                if w not in freq_dict:
                    freq_dict[w] = 1
                else:
                    freq_dict[w] += 1
            print(freq_dict)
            return(freq_dict)

        else:
            return None


    freq_dict = calculate_frequencies(no_stop_words)


    def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> list[str]:
        if isinstance(frequencies, dict) and isinstance(top, int):
            val_lst = sorted(frequencies.values(), reverse=True)
            top_words = []
            counter = 0
            for i in range(top):
                search_freq = val_lst[counter]
                counter += 1
                top_words += [word for word, freq in frequencies.items() if freq == search_freq]
            print(val_lst)
            print(top_words)
        else:
            return None


    top_words = get_top_n(freq_dict, 10)



    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    #assert RESULT, 'Keywords are not extracted'