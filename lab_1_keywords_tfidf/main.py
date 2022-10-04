"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any, Type
from math import log


def check_type_and_emptiness(check_object: Any, check_type: Type[Any], can_be_empty: bool = True) -> bool:
    """
    Checks if object's type matches required type.
    Checks if the object is empty if necessary.
    """
    if not isinstance(check_object, check_type):
        return False
    if not check_object and not can_be_empty:
        return False
    return True


def check_list(check_object: Any, token_type: Type[Any], can_be_empty: bool = True) -> bool:
    """
    Checks if an object is a list or a dictionary and checks types of its elements.
    Checks if the object is empty if necessary.
    """
    if not check_type_and_emptiness(check_object, list, can_be_empty):
        return False
    for i in check_object:
        if not isinstance(i, token_type):
            return False
    return True


def check_dict(check_object: Any, key_type: Type[Any], value_type: Type[Any],
               can_be_empty: bool = True) -> bool:
    """
    Checks if an object is a dictionary and checks types of its keys and values.
    Checks if the object is empty if necessary.
    """
    if not check_type_and_emptiness(check_object, dict, can_be_empty):
        return False
    for k, v in check_object.items():
        if not isinstance(k, key_type) or not isinstance(v, value_type):
            return False
    return True


def check_int(check_object: Any) -> bool:
    """
    Checks if an object is a positive integer.
    """
    if not isinstance(check_object, int) or isinstance(check_object, bool):
        return False
    if not check_object > 0:
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
    text = text.lower().strip()
    res_text = ''
    for i in text:
        if i.isalnum() or i == ' ' or i == '\n':
            res_text += i
    tokens = res_text.replace('\n', ' ').split()
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
    if not check_list(tokens, str, can_be_empty=False) or not check_list(
            stop_words, str):
        return None
    res_tokens = []
    for i in tokens:
        if i not in stop_words:
            res_tokens.append(i)
    return res_tokens


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not check_list(tokens, str, can_be_empty=False):
        return None
    frequencies = {}
    for i in tokens:
        if i not in frequencies.keys():
            frequencies[i] = 1
        else:
            frequencies[i] += 1
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
    # noinspection PyTypeChecker
    if not check_dict(frequencies, str, (int, float), False) or not check_int(top):
        return None
    top_n = sorted(frequencies.keys(), key=lambda x: frequencies[x], reverse=True)[:top]
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
    for i in frequencies.keys():
        token_tf = frequencies[i] / sum(frequencies.values())
        tf_dict[i] = token_tf
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
    if not check_dict(term_freq, str, float, False) or not check_dict(idf, str, float):
        return None
    tfidf = {}
    for i in term_freq.keys():
        if i in idf.keys():
            token_idf = idf[i]
        else:
            token_idf = log(47)
        token_tfidf = term_freq[i] * token_idf
        tfidf[i] = token_tfidf
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
    if not check_dict(doc_freqs, str, int, False) or not check_dict(corpus_freqs, str, int):
        return None
    expected = {}
    for i in doc_freqs.keys():
        j_token = doc_freqs[i]
        if i in corpus_freqs.keys():
            k_token = corpus_freqs[i]
        else:
            k_token = 0
        l_token = sum(doc_freqs.values()) - j_token
        m_token = sum(corpus_freqs.values()) - k_token
        expected_token = ((j_token + k_token) * (j_token + l_token)) / (j_token + k_token + l_token + m_token)
        expected[i] = expected_token
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
    if not check_dict(expected, str, float, False) or not check_dict(observed, str, int, False):
        return None
    chi_values = {}
    for i in expected.keys():
        chi = ((observed[i] - expected[i]) ** 2) / expected[i]
        chi_values[i] = chi
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
    if not check_dict(chi_values, str, float, False) or not isinstance(alpha, float):
        return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion.keys():
        return None
    significant_words = {}
    for i in chi_values.keys():
        if chi_values[i] >= criterion[alpha]:
            significant_words[i] = chi_values[i]
    return significant_words
