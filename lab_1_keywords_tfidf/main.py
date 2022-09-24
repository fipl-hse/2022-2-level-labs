"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any
from math import log


def check_list_and_dict(check_object: Any, object_type: Any, token_type: Any, value_type: Any = None,
                        can_be_empty: bool = True) -> bool:
    """
    Checks if an object is a list or a dictionary and checks types of its elements.
    Checks if the object is empty if necessary.
    """
    if not isinstance(check_object, object_type):
        return False
    if not check_object and not can_be_empty:
        return False
    for i in check_object:
        if not isinstance(i, token_type):
            return False
    if object_type == dict:
        for i in check_object.values():
            if not isinstance(i, value_type):
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
    if isinstance(text, str) and len(text) != 0:
        text = text.lower().strip()
        res_text = ''
        for i in text:
            if i.isalnum() or i == ' ' or i == '\n':
                res_text += i
        tokens = res_text.replace('\n', ' ').split()
        return tokens
    return None


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
    if check_list_and_dict(tokens, list, str, can_be_empty=False) and check_list_and_dict(
            stop_words, list, str):
        res_tokens = []
        for i in tokens:
            if i not in stop_words:
                res_tokens.append(i)
        return res_tokens
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
    if check_list_and_dict(tokens, list, str, can_be_empty=False):
        frequencies = {}
        for i in tokens:
            if i not in frequencies.keys():
                frequencies[i] = 1
            else:
                frequencies[i] += 1
        return frequencies
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
    if check_list_and_dict(frequencies, dict, str, (int, float), False) and check_int(top):
        top_n = sorted(frequencies.keys(), key=lambda x: frequencies[x], reverse=True)[:top]
        return top_n
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
    if check_list_and_dict(frequencies, dict, str, int, False):
        tf_dict = {}
        nd_freq = sum(frequencies.values())
        for i in frequencies.keys():
            nt_freq = frequencies[i]
            token_tf = nt_freq / nd_freq
            tf_dict[i] = token_tf
        return tf_dict
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
    if check_list_and_dict(term_freq, dict, str, float, False) and check_list_and_dict(
            idf, dict, str, float):
        tfidf = {}
        for i in term_freq.keys():
            token_tf = term_freq[i]
            if i in idf.keys():
                token_idf = idf[i]
            else:
                token_idf = log(47 / (0 + 1))
            token_tfidf = token_tf * token_idf
            tfidf[i] = token_tfidf
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
    if check_list_and_dict(doc_freqs, dict, str, int, False) and check_list_and_dict(
            corpus_freqs, dict, str, int):
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
    if check_list_and_dict(expected, dict, str, float, False) and check_list_and_dict(
            observed, dict, str, int, False):
        chi_values = {}
        for i in expected.keys():
            expected_token = expected[i]
            observed_token = observed[i]
            chi = ((observed_token - expected_token) ** 2) / expected_token
            chi_values[i] = chi
        return chi_values
    return None


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
    if check_list_and_dict(chi_values, dict, str, float, False) and isinstance(
            alpha, float):
        criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
        if alpha not in criterion.keys():
            return None
        alpha_criterion = criterion[alpha]
        significant_words = {}
        for i in chi_values.keys():
            token_key = chi_values[i]
            if token_key >= alpha_criterion:
                significant_words[i] = token_key
        return significant_words
    return None
