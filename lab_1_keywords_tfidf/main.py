"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import math


def check_input_type(check_arg, check_type, check_token=None, check_value=None, can_be_empty=True):
    """
    Function that returns None in case of bad input for everything but top
    """
    if not isinstance(check_arg, check_type):
        return False
    if not check_arg and not can_be_empty:
        return False
    if check_type == list:
        for i in check_arg:
            if not isinstance(i, check_token):
                return False
    if check_type == dict:
        for i in check_arg.values():
            if not isinstance(i, check_value):
                return False
        for i in check_arg.keys():
            if not isinstance(i, check_token):
                return None
    return True


def check_num(num_arg):
    """
    Checks top specifically
    """
    if not isinstance(num_arg, int) or isinstance(num_arg, bool):
        return False
    if not num_arg > 0:
        return False
    return True


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    if not check_input_type(text, str):
        return None
    text_small = text.lower()
    tokens_str = ''
    for i in text_small:
        if i.isalnum() is True or i == ' ' or i == '\n':
            tokens_str += i
    tokens = tokens_str.split()
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
    if not check_input_type(tokens, list, str, False) or \
            not check_input_type(stop_words, list, str):
        return None
    no_stop_words = []
    for word in tokens:
        if word not in stop_words:
            no_stop_words.append(word)
    return no_stop_words


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not check_input_type(tokens, list, str, False):
        return None
    tokens_freqs = {}
    for i in tokens:
        tokens_freqs[i] = tokens.count(i)
    return tokens_freqs


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
    if not check_input_type(frequencies, dict, str, (int, float), False) \
            or not check_num(top):
        return None
    sorted_tokens_freqs = {key: value for key, value in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
    top_freqs_words = list(sorted_tokens_freqs.keys())[:top]
    return top_freqs_words


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
    if not check_input_type(frequencies, dict, str, int, False):
        return None
    term_freq = {}
    for key, value in frequencies.items():
        new_v = value / sum(frequencies.values())
        term_freq[key] = new_v
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
    if not check_input_type(term_freq, dict, str, float, False) \
            or not check_input_type(idf, dict, str, float, True):
        return None
    for key in term_freq:
        if key in idf.keys():
            tf_idf_v = idf[key] * term_freq[key]
            term_freq.update({key: tf_idf_v})
        else:
            tf_idf_v = math.log(47/1) * term_freq[key]
            term_freq.update({key: tf_idf_v})
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
