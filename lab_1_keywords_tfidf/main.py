"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any
from string import punctuation
from math import log


def check_list(object_for_check: Any, token_type: type, can_be_empty: bool) -> bool:
    """
    Checks if an object is a list, checks its tokens and its emptiness.
    """
    if not isinstance(object_for_check, list):
        return False
    if not object_for_check and can_be_empty is False:
        return False
    for token in object_for_check:
        if not isinstance(token, token_type):
            return False
    return True


def check_dict(object_for_check: dict, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    """
    Checks if an object is a dictionary, checks its keys and values and its emptiness.
    """
    if not isinstance(object_for_check, dict):
        return False
    if not object_for_check and can_be_empty is False:
        return False
    for key, value in object_for_check.items():
        if not (isinstance(key, key_type) and isinstance(value, value_type)):
            return False
    return True


def check_int(object_for_check: Any) -> bool:
    """
    Checks if an object is an integer and not bool (for get_top_n).
    """
    if not isinstance(object_for_check, int) or isinstance(object_for_check, bool):
        return False
    if object_for_check <= 0:
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
    if not isinstance(text, str) or len(text) == 0:
        return None
    clean_text = ''
    text = text.lower().strip().replace('\n', ' ')
    for token in text:
        if token not in punctuation:
            clean_text += token
    clean_text = clean_text.split()
    return clean_text


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
    if not check_list(tokens, str, False) or not check_list(stop_words, str, True):
        return None
    clean_tokens = []
    for token in tokens:
        if token not in stop_words:
            clean_tokens.append(token)
    return clean_tokens


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not check_list(tokens, str, False):
        return None
    fr_dict = {}
    for token in tokens:
        fr_dict[token] = tokens.count(token)
    return fr_dict


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
    if not check_dict(frequencies, str, Union[int, float], False) or not check_int(top):
        return None
    top_n = sorted(frequencies.keys(), key=lambda token: frequencies[token], reverse=True)[:top]
    return top_n


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
    if not check_dict(frequencies, str, int, False):
        return None
    tf_dict = {}
    for token, frequency in frequencies.items():
        tf_dict[token] = frequency / sum(frequencies.values())
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
    if not check_dict(term_freq, str, float, False) \
            or not check_dict(idf, str, float, True):
        return None
    tfidf_dict = {}
    for token in term_freq.keys():
        if token in idf.keys():
            tfidf_dict[token] = term_freq[token] * idf[token]
        else:
            tfidf_dict[token] = term_freq[token] * log(47)
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
    if not check_dict(doc_freqs, str, int, False) \
            or not check_dict(corpus_freqs, str, int, True):
        return None
    exp_freq_dict = {}
    for token, value in doc_freqs.items():
        if token in corpus_freqs.keys():
            value_corpus = corpus_freqs[token]
        else:
            value_corpus = 0
        all_doc = sum(doc_freqs.values()) - value
        all_corpus = sum(corpus_freqs.values()) - value_corpus
        exp_freq_dict[token] = (value + value_corpus) * (value + all_doc) / (
                value + value_corpus + all_doc + all_corpus)
    return exp_freq_dict


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
    if not check_dict(expected, str, float, False) \
            or not check_dict(observed, str, int, False):
        return None
    chi_dict = {}
    for token, value in expected.items():
        value_observed = observed[token]
        if token not in observed:
            return None
        chi_dict[token] = (value_observed - value)**2 / value
    return chi_dict


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
    if not check_dict(chi_values, str, float, False) or not isinstance(alpha, float):
        return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha in criterion.keys():
        alpha_value = criterion[alpha]
    else:
        return None
    significant_words = {}
    for token, value in chi_values.items():
        if value >= alpha_value:
            significant_words[token] = value
    return significant_words
