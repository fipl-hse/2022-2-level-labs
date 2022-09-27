"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import string
import math


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt arguments, None is returned
    """
    if isinstance(text, str):
        for i in text:
            if i in string.punctuation:
                text = text.replace(i, '')
        return text.lower().split()
    return None


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    """
    Excludes stop words from the token sequence

    Parameters:
    tokens (List[str]): Original token sequence
    stop_words (List[str]: Tokens to exclude

    Returns:
    List[str]: Token sequence that does not include stop words

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(stop_words, list) or not isinstance(tokens, list):
        return None
    for i in tokens:
        if not isinstance(i, str):
            return None
    for i in stop_words:
        if not isinstance(i, str):
            return None
    clean_tokens = []
    for i in tokens:
        if i not in stop_words:
            clean_tokens.append(i)
    return clean_tokens


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    dictionary = dict()
    if not isinstance(tokens, list) or tokens == []:
        return None
    for i in tokens:
        if not isinstance(i, str):
            return None
        number = tokens.count(i)
        dictionary[i] = number
    return dictionary


def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
    """
    Extracts a certain number of most frequent tokens

    Parameters:
    frequencies (Dict): A dictionary with tokens and
    its corresponding frequency values
    top (int): Number of token to extract

    Returns:
    List[str]: Sequence of specified length
    consisting of tokens with the largest frequency

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(frequencies, dict) or not isinstance(top, int) or isinstance(top, bool) or frequencies == {}:
        return None
    if top <= 0:
        return None
    for i in frequencies.items():
        if not isinstance(i[0], str) or not isinstance(i[1], (int, float)):
            return None
    sorted_list = []
    sort = sorted(frequencies.items(), reverse=True, key=lambda x: x[1])
    for i in sort:
        sorted_list.append(i[0])
    if top > len(frequencies):
        top = len(frequencies)
    return sorted_list[:top]


def calculate_tf(frequencies: dict[str, int]) -> Optional[dict[str, float]]:
    """
    Calculates Term Frequency score for each word in a token sequence
    based on the raw frequency

    Parameters:
    frequencies (Dict): Raw number of occurrences for each of the tokens

    Returns:
    dict: A dictionary with tokens and corresponding term frequency score

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(frequencies, dict):
        return None
    for i in frequencies.items():
        if not isinstance(i[0], str) or not isinstance(i[1], int):
            return None
    dictionary = {}
    all_instances = sum(frequencies.values())
    for i in frequencies.keys():
        dictionary[i] = frequencies[i] / int(all_instances)
    return dictionary


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
    if not isinstance(term_freq, dict) or not isinstance(idf, dict) or term_freq == {}:
        return None
    for i in term_freq.items():
        if not isinstance(i[0], str) or not isinstance(i[1], float):
            return None
    for i in idf.items():
        if not isinstance(i[0], str) or not isinstance(i[1], float):
            return None
    dictionary = {}
    for i in term_freq.keys():
        if i in idf.keys():
            tfidf = term_freq[i] * idf[i]
        else:
            tfidf = term_freq[i] * math.log(47 / (0 + 1))
        dictionary[i] = tfidf
    return dictionary


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
    have a chi-squared value greater than the criterion

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
