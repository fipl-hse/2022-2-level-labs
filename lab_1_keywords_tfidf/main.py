"""
Lab 1
Extract keywords based on frequency related metrics
"""

from typing import Optional, Union
import string
import math


def dictionary_check(dictionary: dict, possible_type: type, empty: bool) -> bool:

    """
    Сheck the correctness of a dictionary
    And its elements
    """
    if isinstance(dictionary, dict):
        if dictionary == {} and not empty:
            return False
        for key, value in dictionary.items():
            if not isinstance(key, str) or not isinstance(value, (int, possible_type)) or isinstance(value, bool):
                return False
        return True
    return False


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    if not text or not isinstance(text, str):
        return None
    text = text.lower()
    for element in string.punctuation:
        text = text.replace(element, '')
    tokens = text.split()
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
    if not tokens or not isinstance(tokens, list):
        return None
    tokens_clean = [i for i in tokens if i not in stop_words]
    return tokens_clean


def calculate_frequencies(tokens_clean: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens_clean, list):
        return None
    for word in tokens_clean:
        if isinstance(word, str):
            frequencies = {i: tokens_clean.count(i) for i in tokens_clean}
            return frequencies
    return None


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
    if not dictionary_check(frequencies, float, False)\
            or not (not isinstance(top, bool) and isinstance(top, int) and not top <= 0):
        return None
    words = sorted(frequencies.keys(), key=lambda key: frequencies[key], reverse=True)
    top_five = words[:top]
    return top_five


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
    if not dictionary_check(frequencies, int, True):
        return None
    term_dict = {word: (numb / sum(frequencies.values())) for word, numb in frequencies.items()}
    return term_dict


def calculate_tfidf(term_dict: dict[str, float], idf: dict[str, float]) -> Optional[dict[str, float]]:
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
    if not dictionary_check(term_dict, float, False) or not isinstance(idf, dict):
        return None
    tfidf_dict = {}
    for word in term_dict:
        if word not in idf.keys():
            idf[word] = math.log(47/1)
        tfidf_dict[word] = term_dict[word] * idf[word]
    return tfidf_dict


def calculate_expected_frequency(doc_freqs: dict[str, int], corpus_freqs: dict[str, int]) -> Optional[dict[str, float]]:
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
