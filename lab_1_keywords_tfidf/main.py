"""
Lab 1
Extract keywords based on frequency related metrics
"""
import string
from math import log
from typing import Optional, Union, Any


def check_types(value: Any, keys_type: type, values_type: Union[type, tuple], can_empty: bool = False) -> bool:
    """
    Raises ValueError when type is not expected
    :param value:
    :param keys_type:
    :param values_type:
    :param can_empty:
    """
    if not can_empty and not value:
        return False
    if not isinstance(value, dict):
        return False
    for element in value:
        if not isinstance(element, keys_type):
            return False
    for element in value.values():
        if not isinstance(element, values_type):
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

    text = text.replace('-', ' ').lower().strip()
    for excess_symbol in string.punctuation:
        text = text.replace(excess_symbol, '')
    return text.split()


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
    if not (isinstance(tokens, list) and isinstance(stop_words, list)):
        return None

    return [word for word in tokens if word not in stop_words]


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) or not tokens:
        return None
    for element in tokens:
        if not isinstance(element, str):
            return None

    frequency_dict = {}
    for word in tokens:
        frequency_dict[word] = frequency_dict.get(word, 0) + 1
    return frequency_dict


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
    if not check_types(frequencies, str, (int, float)) or not isinstance(top, int) or top <= 0 or isinstance(top, bool):
        return None

    return sorted(frequencies, key=lambda word: frequencies[word], reverse=True)[:top]


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
    if not check_types(frequencies, str, int):
        return None

    quantity = sum(frequencies.values())
    return {word: freq / quantity for word, freq in frequencies.items()}


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
    if not check_types(term_freq, str, float) or not check_types(idf, str, float, can_empty=True):
        return None

    return {word: freq * idf.get(word, log(47)) for word, freq in term_freq.items()}


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
    if not check_types(doc_freqs, str, int) or not check_types(corpus_freqs, str, int, can_empty=True):
        return None

    expected_frequency = {}
    for word, freq in doc_freqs.items():
        corpus_freq = corpus_freqs.get(word, 0)
        except_word_doc_freq = sum(doc_freqs.values()) - freq
        except_word_corpus_freq = sum(corpus_freqs.values()) - corpus_freq
        expected_frequency[word] = ((freq + corpus_freq) * (freq + except_word_doc_freq)) / (
                    freq + corpus_freq + except_word_doc_freq + except_word_corpus_freq)
    return expected_frequency


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
    if not check_types(expected, str, float) or not check_types(observed, str, int):
        return None

    chi_values = {}
    for word, freq in expected.items():
        chi_values[word] = (observed.get(word, 0) - freq) ** 2 / freq
    return chi_values


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
    if not check_types(chi_values, str, float) or not isinstance(alpha, float):
        return None

    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None
    return {word: value for word, value in chi_values.items() if value > criterion[alpha]}
