"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import re
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

    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()

    return tokens


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    """
    Excludes stop words from the token sequence

    Parameters:
    tokens (List[str]): Original token sequence
    stop_words (List[str]): Tokens to exclude

    Returns:
    List[str]: Token sequence that does not include stop words

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) or not isinstance(stop_words, list):
        return None

    for word in stop_words:
        if not isinstance(word, str):
            return None
        while word in tokens:
            tokens.remove(word)

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
    for token in tokens:
        if not isinstance(token, str):
            return None
        frequencies[token] = tokens.count(token)

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
    if not isinstance(frequencies, dict) or not frequencies:
        return None
    if not isinstance(top, int) or top <= 0 or top is (True or False):
        return None

    if top > len(frequencies):
        frequencies_sorted = sorted(frequencies.items(), key=lambda x: -x[1])
    else:
        frequencies_sorted = sorted(frequencies.items(), key=lambda x: -x[1])[:top]
    most_frequent = list(dict(frequencies_sorted))
    return most_frequent


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
    if not isinstance(frequencies, dict) or not frequencies:
        return None
    for i in frequencies.keys():
        if not isinstance(i, str):
            return None

    term_freq = {}
    for token, value in frequencies.items():
        term_freq[token] = value/sum(frequencies.values())

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
    if not isinstance(term_freq, dict) or not isinstance(idf, dict) or not term_freq:
        return None
    for token in term_freq.keys():
        if not isinstance(token, str):
            return None

    doc_freqs = {}
    for token, value in term_freq.items():
        if token in idf:
            doc_freqs[token] = value * idf[token]
        else:
            doc_freqs[token] = value * math.log(47/1)

    return doc_freqs


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
    if not isinstance(doc_freqs, dict) or not doc_freqs:
        return None
    if not isinstance(corpus_freqs, dict):
        return None
    for token_in_doc in doc_freqs.keys():
        if not isinstance(token_in_doc, str):
            return None

    expected = {}
    for token, value in doc_freqs.items():
        if not corpus_freqs:
            expected[token] = value
        else:
            j_doc = doc_freqs[token]
            l_doc = sum(doc_freqs.values()) - j_doc
            if token in corpus_freqs:
                k_corp = corpus_freqs[token]
            else:
                k_corp = 0
            m_corp = sum(corpus_freqs.values()) - k_corp
            expected[token] = ((j_doc + k_corp) * (j_doc + l_doc)) / (j_doc + k_corp + l_doc + m_corp)
    return expected



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
    if not isinstance(expected, dict) or not expected or not isinstance(observed, dict) or not observed:
        return None
    for token in expected.keys():
        if not isinstance(token, str):
            return None
    for token in observed.keys():
        if not isinstance(token, str):
            return None
    for value in expected.values():
        if not isinstance(value, float):
            return None
    for value in observed.values():
        if not isinstance(value, int):
            return None

    chi_value = {}
    for token in observed.keys():
        chi_value[token] = ((observed[token]-expected[token])**2)/expected[token]
    return chi_value



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
    if not isinstance(chi_values, dict) or not chi_values:
        return None
    if not isinstance(alpha, float):
        return None
    for token in chi_values.keys():
        if not isinstance(token, str):
            return None

    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion.keys():
        return None
    significant_words = {}
    for token in chi_values.keys():
        if chi_values[token] >= criterion[alpha]:
            significant_words[token] = chi_values[token]

    return significant_words
