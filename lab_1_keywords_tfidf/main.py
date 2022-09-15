"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import string
import math


def right_type(container: Union[list, tuple, dict, set], container_type: type,
               elements_type: Union[type, dict[str, type]], allow_empty=True) -> bool:
    """
    Checks datatype of a container and its elements
    """
    empty_check = allow_empty or container
    if type(container) == container_type and type(container) == dict:
        return (all([type(key) == elements_type['keys'] for key in container.keys()])
                and all([type(value) == elements_type['values'] for value in container.values()])
                and empty_check)
    if type(container) == container_type and type(container) != dict:
        return all([type(i) == elements_type for i in container]) and empty_check
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
    if type(text) == str:
        without_punctuation = ''.join([i for i in text if i not in string.punctuation])
        tokens = [token for token in without_punctuation.lower().split()]
        return tokens
    return None


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    """
    Excludes stop words from the token sequence

    Parameters:
    tokens List[str]): Original token sequence
    stop_words List[str]: Tokens to exclude

    Returns:
    List[str]: Token sequence that does not include stop words

    In case of corrupt input arguments, None is returned
    """
    if right_type(tokens, list, str) and right_type(stop_words, list, str):
        return [token for token in tokens if token not in stop_words]
    return None


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if right_type(tokens, list, str, allow_empty=False):
        return {token: tokens.count(token) for token in tokens}
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
    if (type(top) == int
            and top > 0
            and (right_type(frequencies, dict, {'keys': str, 'values': int}, allow_empty=False)
                 or right_type(frequencies, dict, {'keys': str, 'values': float}, allow_empty=False))):
        return sorted(frequencies, key=frequencies.get, reverse=True)[:top]
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
    if (type(frequencies) == dict
            and right_type(frequencies, dict, {'keys': str, 'values': int}, allow_empty=False)):
        total = sum(frequencies.values())
        return {term: occur / total for term, occur in frequencies.items()}
    return None


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
    if (right_type(term_freq, dict, {'keys': str, 'values': float}, allow_empty=False)
            and right_type(idf, dict, {'keys': str, 'values': float})):
        tfidf = {}
        for term in term_freq:
            if idf.get(term) is not None:
                tfidf[term] = idf.get(term) * term_freq[term]
            else:
                tfidf[term] = math.log(47 / (0 + 1)) * term_freq[term]
        return tfidf
    return None


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
    if (right_type(doc_freqs, dict, {'keys': str, 'values': int}, allow_empty=False) and
            right_type(corpus_freqs, dict, {'keys': str, 'values': int})):
        new_freq = {}
        for token in doc_freqs:
            a = doc_freqs[token]
            b = corpus_freqs.get(token, 0)
            d = sum(corpus_freqs.values()) - b
            c = sum(doc_freqs.values()) - a
            new_freq[token] = ((a + c) * (a + b)) / (a + c + d + b)
        return new_freq
    return None


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
    if (right_type(expected, dict, {'keys': str, 'values': float}, allow_empty=False)
            and right_type(observed, dict, {'keys': str, 'values': int}, allow_empty=False)):
        new_freq = {}
        for token in expected:
            new_freq[token] = ((observed[token] - expected[token]) ** 2) / expected[token]
        return new_freq
    return None


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
    if (right_type(chi_values, dict, {'keys': str, 'values': float}, allow_empty=False)
            and type(alpha) == float):
        criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
        try:
            return {word: value for word, value in chi_values.items() if value > criterion[alpha]}
        except KeyError:
            pass
    return None
