"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union



def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """

    import re

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
    stop_words (List[str]: Tokens to exclude

    Returns:
    List[str]: Token sequence that does not include stop words

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) or not tokens:
        return None
    if not isinstance(stop_words, list):
        return None

    for stop_word in  stop_words:
        if not isinstance(stop_word, str):
            return None
        while stop_word in tokens:
            tokens.remove(stop_word)

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
    if not isinstance(tokens, list) or not tokens:
        return None

    frequencies = {}
    for token in tokens:
        if not isinstance(token, str):
            return None
        if token not in frequencies.keys():
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

    top_frequencies = dict(sorted(frequencies.items(), reverse=True, key=lambda i: i[1])[:top])

    return list(top_frequencies.keys())


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

    term_frequencies = {}
    num_of_tokens = sum(frequencies.values())

    for token in frequencies.keys():
        if not isinstance(token, str):
            return None
        term_frequencies[token] = frequencies[token]/num_of_tokens

    return term_frequencies


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
    if not isinstance(term_freq, dict) or not term_freq:
        return None
    if not isinstance(idf, dict):
        return None

    tf_idf = {}

    from math import log

    for token in term_freq:
        if not isinstance(token, str):
            return None
        if token in idf:
            tf_idf[token] = term_freq[token] * idf[token]
        else:
            tf_idf[token] = term_freq[token] * log(47)

    return tf_idf


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

    expected_freqs = {}
    for token in doc_freqs.keys():
        if not isinstance(token, str):
            return None
        j = doc_freqs[token]
        if token in corpus_freqs.keys():
            k = corpus_freqs[token]
        else:
            k = 0
        l = sum(doc_freqs.values()) - j
        m = sum(corpus_freqs.values()) - k
        expected_freqs[token] = (j+k)*(j+l)/(j+k+l+m)

    return expected_freqs


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
    if not isinstance(expected, dict) or not expected:
        return None
    if not isinstance(observed, dict) or not observed:
        return None
    for token in observed.keys():
        if not isinstance(token, str):
            return None
    for i in observed.values():
        if not isinstance(i, int):
            return None
    for i in expected.values():
        if not isinstance(i, float):
            return None

    chi_values = {}

    for token in expected.keys():
        if not isinstance(token, str):
            return None
        chi_values[token] = (observed[token]-expected[token])**2 / expected[token]

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
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if not isinstance(chi_values, dict) or not chi_values:
        return None
    if not isinstance(alpha, float) or not alpha or alpha not in criterion.keys():
        return None

    significant_words = {}

    for token in chi_values:
        if not isinstance(token, str):
            return None
        if not isinstance(chi_values[token], float):
            return None
        if chi_values[token] >= criterion[alpha]:
            significant_words[token] = chi_values[token]

    return significant_words
