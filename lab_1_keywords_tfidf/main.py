"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union

import json
from pathlib import Path

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
        punctuation = '!?-.,\'\"():;@#№$%^<>&*/`~|'
        for e in text:
            if e not in punctuation:
                no_punc_text += e
        all_words = no_punc_text.lower().strip().split()
        return all_words
    else:
        return None


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    if isinstance(tokens, list) and isinstance(stop_words, list):
        no_stop_words = [w for w in tokens if w not in stop_words]
        return no_stop_words
    else:
        return None


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    if (isinstance(tokens, list) and tokens
            and all(isinstance(w, str) for w in tokens)):
        freq_dict = {}
        for w in tokens:
            if w not in freq_dict:
                freq_dict[w] = 1
            else:
                freq_dict[w] += 1
        return freq_dict

    else:
        return None


def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
    if (isinstance(frequencies, dict) and frequencies
            and all(isinstance(k, str) for k in frequencies.keys())
            and all(isinstance(v, int or float) for v in frequencies.values())
            and isinstance(top, int) and type(top) != bool and top > 0):

        val_lst = sorted(frequencies.values(), reverse=True)
        top_words = []
        counter = 0

        if top > len(val_lst):
            for v in val_lst:
                search_freq = val_lst[counter]
                counter += 1
                top_words += [word for word, freq in frequencies.items()
                              if freq == search_freq and word not in top_words]

        else:
            for i in range(top):
                search_freq = val_lst[counter]
                counter += 1
                top_words += [word for word, freq in frequencies.items()
                              if freq == search_freq and word not in top_words]
        while len(top_words) > top:
            top_words.pop()

        return top_words

    else:
        return None


def calculate_tf(frequencies: dict[str, int]) -> Optional[dict[str, float]]:
    if isinstance(frequencies, dict) and frequencies.keys() and frequencies.values():
        all_words = sum(frequencies.values())
        tf_dict = {}
        for w, f in frequencies.items():
            tf = f / all_words
            tf_dict[w] = tf
        return tf_dict
    else:
        None


def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> Optional[dict[str, float]]:
    """
    Calculates TF-IDF score for each of the tokens
    based on its TF and IDF scores

    Parameters:
    term_freq (Dict): A dictionary with tokens and its corresponding TF values
    idf (Dict): A dictionary with tokens and its corresponding IDF values

    Returns:
    Dict: A dictionary with tokens and its corresponding TF-IDF values

    In case of corrupt input arguments, None is returned
    """
    pass


def calculate_expected_frequency(
    doc_freqs: dict[str, int], corpus_freqs: dict[str, int]
) -> Optional[dict[str, float]]:
    """
    Calculates expected frequency for each of the tokens based on its
    Term Frequency score for both target document and general corpus

    Parameters:
    doc_freqs (Dict): A dictionary with tokens and its corresponding number of occurrences in document
    corpus_freqs (Dict): A dictionary with tokens and its corresponding number of occurrences in corpus

    Returns:
    Dict: A dictionary with tokens and its corresponding expected frequency

    In case of corrupt input arguments, None is returned
    """
    pass


def calculate_chi_values(expected: dict[str, float], observed: dict[str, int]) -> Optional[dict[str, float]]:
    """
    Calculates chi-squared value for the tokens
    based on their expected and observed frequency rates

    Parameters:
    expected (Dict): A dictionary with tokens and
    its corresponding expected frequency
    observed (Dict): A dictionary with tokens and
    its corresponding observed frequency

    Returns:
    Dict: A dictionary with tokens and its corresponding chi-squared value

    In case of corrupt input arguments, None is returned
    """
    pass


def extract_significant_words(chi_values: dict[str, float], alpha: float) -> Optional[dict[str, float]]:
    """
    Select those tokens from the token sequence that
    have a chi-squared value smaller than the criterion

    Parameters:
    chi_values (Dict): A dictionary with tokens and
    its corresponding chi-squared value
    alpha (float): Level of significance that controls critical value of chi-squared metric

    Returns:
    Dict: A dictionary with significant tokens
    and its corresponding chi-squared value

    In case of corrupt input arguments, None is returned
    """
    pass
