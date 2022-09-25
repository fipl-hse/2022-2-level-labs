"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import math


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str):
        return None
    text = text.replace('\n', ' ').lower().strip()
    fin_text = ''
    for symbol in text:
        if symbol.isalnum() is True or symbol == ' ':
            fin_text += symbol
    tokens = fin_text.split()
    return tokens


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
    if not isinstance(tokens, list) or not isinstance(stop_words, list):
        return None
    for i in tokens.copy():
        if not isinstance(i, str):
            return None
        else:
            if i in stop_words:
                tokens.remove(i)

    return tokens


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list):
        return None

    frequencies = {}
    for i in tokens:
        if not isinstance(i, str):
            return None
        else:
            freq = tokens.count(i)
            frequencies[i] = freq
    return frequencies


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

    if not isinstance(frequencies, dict) or not isinstance(top, int) or isinstance(top, bool):
        return None

    for k, v in frequencies.items():
        if not isinstance(k, str) or not isinstance(v, int | float):  # or isinstance(v, bool):
            return None
        else:
            sorted_keys = sorted(frequencies, reverse = True, key=lambda word: frequencies[word])
            if not sorted_keys:
                return None
            else:
                if top > 0:
                    return sorted_keys[:top]
                else:
                    return None


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

    term_freq = {}
    length = 0

    for k, v in frequencies.items():
        if not isinstance(k, str) or not isinstance(v, int):
            return None
        else:
            length = length + v

    for k, v in frequencies.items():
        tf = v / length
        term_freq[k] = tf

    return term_freq



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
    if not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None
    if term_freq == {}:
        return None

    for k, v in term_freq.items():
        if not isinstance(k, str) or not isinstance(v, float):
            return None
        if idf == {}:
            term_freq[k] = v * math.log(47 / 1)
        else:
            for k1, v1 in idf.items():
                if not isinstance(k1, str) or not isinstance(v1, float):
                    return None
                else:
                    if k == k1:
                        term_freq[k] = v * idf[k]
                        break
                    else:
                        term_freq[k] = v * math.log(47 / 1)

    return term_freq



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
