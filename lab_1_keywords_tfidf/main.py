"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any
from math import log
from string import punctuation


def correct_dict(variable: dict, type1: type, type2: type, empty: bool) -> bool:
    """
    Checks the type of dict, keys and values
    """
    if isinstance(variable, dict):
        if not empty and not variable:
            return False
        for key, value in variable.items():
            if not (isinstance(key, type1) or isinstance(value, type2)):
                return False
        return True
    return False


def correct_list(variable: list, type1: type, empty: bool) -> bool:
    """
    Checks the type of list and its elements
    """
    if isinstance(variable, list):
        if not empty and not variable:
            return False
        for i in variable:
            if not isinstance(i, type1):
                return False
        return True
    return False


def is_int(variable: Any) -> bool:
    """
    Checks whether type of variable is int
    """
    if isinstance(variable, int):
        if not isinstance(variable, bool):
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
    if not isinstance(text, str):
        return None
    new_text = ""
    for symbol in text.lower():
        if symbol not in punctuation:
            new_text += symbol
    split_words = new_text.split()
    return split_words


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
    if not (correct_list(tokens, str, False) and correct_list(stop_words, str, True)):
        return None
    not_stop_words = []
    for token in tokens:
        if token not in stop_words:
            not_stop_words.append(token)
    return not_stop_words


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not correct_list(tokens, str, False):
        return None
    freq_dict = {}
    for token in tokens:
        if token in freq_dict:
            freq_dict[token] += 1
        else:
            freq_dict[token] = 1
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
    correct = correct_dict(frequencies, str, float, False) or correct_dict(frequencies, str, int, False)
    if not (is_int(top) and top > 0 and correct):
        return None
    return sorted(frequencies.keys(), key=lambda key: frequencies[key], reverse=True)[:top]


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
    if not correct_dict(frequencies, str, int, False):
        return None
    tf_dict = {}
    for key, value in frequencies.items():
        tf_dict[key] = value / sum(frequencies.values())
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
    if not (correct_dict(term_freq, str, float, False) and correct_dict(idf, str, float, True)):
        return None
    tfidf = {}
    for key in term_freq.keys():
        try:
            tfidf[key] = term_freq[key] * idf[key]
        except KeyError:
            tfidf[key] = log(47 / (0 + 1)) * term_freq[key]
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
    if not (correct_dict(doc_freqs, str, int, False) and correct_dict(corpus_freqs, str, int, True)):
        return None
    expected_dict = {}
    for key, value in doc_freqs.items():
        number_words_corpus = corpus_freqs.get(key, 0)
        number_other_words_corpus = sum(corpus_freqs.values()) - number_words_corpus
        number_other_words_doc = sum(doc_freqs.values()) - value
        expected_dict[key] = ((value + number_words_corpus) * (value + number_other_words_doc)) / (
                value + number_words_corpus + number_other_words_doc + number_other_words_corpus
        )
    return expected_dict


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
    if not (correct_dict(expected, str, float, False) and correct_dict(observed, str, int, False)):
        return None
    x2_dict = {}
    for key, value in expected.items():
        x2_dict[key] = (observed.get(key, 0) - value) ** 2 / value
    return x2_dict


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
    if not (correct_dict(chi_values, str, float, False) and isinstance(alpha, float) and alpha in criterion.keys()):
        return None
    significant_words_dict = {}
    for key, value in chi_values.items():
        if value > criterion[alpha]:
            significant_words_dict[key] = value
    return significant_words_dict
