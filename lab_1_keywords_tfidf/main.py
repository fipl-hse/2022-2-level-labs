"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union

import math
import string


def dict_type_check(dictionary: dict, key_type: Union[type, tuple], val_type: Union[type, tuple]) -> bool:
    """
    Checks whether the object is a dictionary and its keys and values are of the expected type

    Parameters:
    dictionary (Dict): any dictionary which keys and values types must be checked
    key_type (Type or tuple): the expected key type
    val_type (Type or tuple): the expected value type

    Returns:
    bool: True if dictionary keys and values are of the expected type, False otherwise
    """
    if not(isinstance(dictionary, dict)
            and all(isinstance(key, key_type) for key in dictionary.keys())
            and all(isinstance(value, val_type) for value in dictionary.values())):
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
    if not isinstance(text, str):
        return None
    for punc in string.punctuation:
        text = text.replace(punc, '')
    clean_words = text.lower().split()
    return clean_words


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
    if not(tokens and isinstance(tokens, list) and all(isinstance(token, str) for token in tokens)
            and isinstance(stop_words, list)):
        return None
    no_stop_words = [token for token in tokens if token not in stop_words]
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
    if not(tokens and isinstance(tokens, list)
            and all(isinstance(token, str) for token in tokens)):
        return None
    freq_dict = {token: tokens.count(token) for token in tokens}
    return freq_dict


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
    if not(frequencies and dict_type_check(frequencies, str, (int, float))
            and isinstance(top, int) and top is not (True or False) and top > 0):
        return None
    sorted_words = [token for token, freq in sorted(frequencies.items(), key=lambda token: token[1], reverse=True)]
    top_words = sorted_words[:top]
    return top_words


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
    if not(frequencies and dict_type_check(frequencies, str, int)):
        return None
    words_num = sum(frequencies.values())
    tf_dict = {token: (freq / words_num) for token, freq in frequencies.items()}
    return tf_dict


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
    if not(term_freq and dict_type_check(term_freq, str, float)
            and dict_type_check(idf, str, float)):
        return None
    tfidf_dict = {token: (term_freq[token] * idf.get(token, math.log(47/1))) for token in term_freq}
    return tfidf_dict


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
    if not(doc_freqs and dict_type_check(doc_freqs, str, int)
            and dict_type_check(corpus_freqs, str, int)):
        return None

    expected_freq_dict = {}

    for token, freq in doc_freqs.items():

        collection_freq = corpus_freqs.get(token, 0)
        partial_doc_words_sum = sum(doc_freqs.values()) - freq
        partial_collection_words_sum = sum(corpus_freqs.values()) - collection_freq

        expected = (((freq + collection_freq) * (freq + partial_doc_words_sum)) /
                    (freq + collection_freq + partial_doc_words_sum + partial_collection_words_sum))

        expected_freq_dict[token] = expected
    return expected_freq_dict


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
    if not(expected and dict_type_check(expected, str, float)
            and observed and dict_type_check(observed, str, int)):
        return None
    chi_val_dict = {token: (((freq - expected[token]) ** 2) / (expected[token]))
                    for token, freq in observed.items()}
    return chi_val_dict


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
    if not(chi_values and dict_type_check(chi_values, str, float) and alpha in [0.05, 0.01, 0.001]):
        return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    significant_chi_words = {token: chi_val for token, chi_val in chi_values.items() if chi_val >= criterion[alpha]}
    return significant_chi_words
