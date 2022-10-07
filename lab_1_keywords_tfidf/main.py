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
    text = text.lower().strip()
    punctuation_of_the_text = """.,!?""“”'':;_()-[]{}\|/`~#№$%&*@=+<>"""
    for p in punctuation_of_the_text:
        text = text.replace(p, '')
    text = text.split()
    return text


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
    for s_w in stop_words:
        if s_w in tokens:
            while s_w in tokens:
                tokens.remove(s_w)
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
    for symbol in tokens:
        if not isinstance(symbol, str):
            return None
    frequency_dictionary = {}
    for word in tokens:
        frequency_dictionary[word] = 1 if word not in frequency_dictionary else frequency_dictionary[word] + 1
    return frequency_dictionary


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

    if not isinstance(frequencies, dict) or not frequencies or not isinstance(top, int) or isinstance(top,
                                                                                                      bool) or top <= 0:
        return None
    if top > len(frequencies):
        get_sorted_n = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    else:
        get_sorted_n = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:top]
    n_top = list(dict(get_sorted_n))
    return n_top


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
    for word in frequencies.keys():
        if not isinstance(word, str):
            return None
    dict_tf = {}
    for token, count in frequencies.items():
        dict_tf[token] = count / sum(frequencies.values())
    return dict_tf


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
    for word in term_freq.keys():
        if not isinstance(word, str):
            return None
    tfidf = {}
    for worrd, va1ue in term_freq.items():
        if worrd in idf:
            tfidf[worrd] = va1ue * idf[worrd]
        else:
            tfidf[worrd] = va1ue * math.log(47 / 1)
    return tfidf


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
