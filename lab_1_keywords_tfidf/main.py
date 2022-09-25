"""
Lab 1
Extract keywords based on frequency related metrics
"""
import string
from typing import Optional, Union, Any
import math


def check(obj: Any, exp_type: Any, exp_cont: Any = None, exp_val: Any = None, not_empty: bool = False) -> bool:
    """
    Checks any type used in program. Also works for types of containers' content.

    Parameters:
    obj (Any): An object which type is checked
    exp_type (Any): A type we expect obj to be
    exp_cont (Any): A type we expect the content (elements for lists or keys for dictionaries) to be (optional)
    exp_val (Any): A type we expect the values in a dictionary to be (optional)
    not_empty (bool): If exp_type is a container, True stands for "it should not be empty" (optional)

    Returns:
    bool: True if obj (and its content if needed) has the expected type, False otherwise
    """
    return not (not isinstance(obj, exp_type) or exp_type == int and isinstance(obj, bool)) \
        and not (exp_type in (list, dict) and not_empty and not obj) \
        and not (exp_type in (list, dict) and exp_cont and not all(check(item, exp_cont) for item in obj)) \
        and not (exp_type == dict and exp_val and not all(check(value, exp_val) for value in obj.values()))


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    if not check(text, str):
        return None
    return ''.join(symbol for symbol in text if symbol not in string.punctuation).lower().split()


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
    if not check(tokens, list, str) or not check(stop_words, list, str):
        return None
    return [token for token in tokens if token not in stop_words]


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not check(tokens, list, str, not_empty=True):
        return None
    return {token: tokens.count(token) for token in tokens}


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
    if not check(frequencies, dict, str, (float, int), True) or not check(top, int) or top <= 0:
        return None
    return sorted(list(frequencies), reverse=True, key=lambda word: frequencies[word])[:top]


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
    if not check(frequencies, dict, str, int):
        return None
    total_words = sum(frequencies.values())
    return {token: frequencies[token] / total_words for token in frequencies}


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
    if not check(term_freq, dict, str, float, True) or not check(idf, dict, str, float):
        return None
    return {key: term_freq[key] * (idf[key] if key in idf else math.log(47)) for key in term_freq}


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
    if not check(doc_freqs, dict, str, int, True) or not check(corpus_freqs, dict, str, int):
        return None
    doc_total, corpus_total = sum(doc_freqs.values()), sum(corpus_freqs.values())
    # j = doc_freq;             k = corpus_freqs[key] (0 if not there)
    # l = doc_total - doc_freq; m = corpus_total - corpus_freqs[key]
    return {key: (doc_freq + (corpus_freqs[key] if key in corpus_freqs else 0)) * doc_total /
                 (doc_total + corpus_total) for key, doc_freq in doc_freqs.items()}  # (j+k)*(j+l)/(j+k+l+m)


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
    if not check(expected, dict, str, float, True) or not check(observed, dict, str, int, True):
        return None
    return {key: (value - expected[key]) ** 2 / expected[key] for key, value in observed.items()}


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
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if not check(chi_values, dict, str, float, True) or not check(alpha, float) or alpha not in criterion:
        return None
    return {key: value for key, value in chi_values.items() if value > criterion[alpha]}
