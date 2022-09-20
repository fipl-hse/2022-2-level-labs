"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any
import math


def check(obj: Any, exp_type: Any, exp_cont: Any = None, exp_val: Any = None, not_empty: bool = False) -> bool:
    """
    Checks any type used in program. Also works for types of containers' content.
    Parameters:
    obj (Any): An object which type is checked
    exp_type (Any): A type we expect obj to be
    exp_cont (Any): A type we expect the content (elements for lists or keys for dictionaries) to be (optional)
    exp_val (Any): A type we expect the values in a dictionary to be (optional)
    not_empty (bool): If exp_type is a container, True stands for "it should not be empty" (optional)
    Returns:
    bool: True if obj (and its content if needed) has the expected type, False otherwise
    """
    flag = True
    if not isinstance(obj, exp_type) or exp_type == int and isinstance(obj, bool):
        flag = False
    if exp_type in (list, dict) and not_empty and not obj and flag:
        flag = False
    if exp_type in (list, dict) and exp_cont and flag:
        for item in obj:
            if not check(item, exp_cont):
                flag = False
    if exp_type == dict and exp_val and flag:
        for value in obj.values():
            if not check(value, exp_val):
                flag = False
    return flag


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    if not check(text, str):
        return None
    text = text.lower()
    for bad_symbol in '.,:-!?;%<>&*@#()':
        text = text.replace(bad_symbol, '')
    text_split = text.split()
    return text_split


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
    if not check(tokens, list) or not check(stop_words, list):
        return None
    for stop_word in stop_words:
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
    if not check(tokens, list, exp_cont=str, not_empty=True):
        return None
    frequency_dict = {}
    for token in tokens:
        if token in frequency_dict.keys():
            frequency_dict[token] += 1
        else:
            frequency_dict[token] = 1
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
    if not check(frequencies, dict, exp_cont=str, exp_val=(float, int), not_empty=True) \
            or not check(top, int) or top <= 0:
        return None
    top_list = []
    for key in frequencies.keys():
        top_list.append(key)
    top_list.sort(reverse=True, key=lambda word: frequencies[word])
    return top_list[:top]


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
    if not check(frequencies, dict, exp_cont=str):
        return None
    tf_dict = {}
    total_words = sum(frequencies.values())
    for token in frequencies.keys():
        tf_dict[token] = frequencies[token] / total_words
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
    if not check(term_freq, dict, exp_cont=str, exp_val=float, not_empty=True) or not check(idf, dict):
        return None
    tfidf_dict = {}
    for key in term_freq.keys():
        if key in idf.keys():
            tfidf_dict[key] = term_freq[key] * idf[key]
        else:
            tfidf_dict[key] = term_freq[key] * math.log(47)
    return tfidf_dict


def calculate_expected_frequency(doc_freqs: dict[str, int], corpus_freqs: dict[str, int]) -> Optional[dict[str, float]]:
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
    if not check(doc_freqs, dict, exp_cont=str, not_empty=True) or not check(corpus_freqs, dict, exp_cont=str):
        return None
    doc_total = sum(doc_freqs.values())
    corpus_total = sum(corpus_freqs.values())
    exp_freqs = {}
    for key, doc_freq in doc_freqs.items():
        # j = doc_freq
        # k = corpus_freqs[key] (0 if does not appear in corpus)
        corpus_freq = 0
        if key in corpus_freqs.keys():
            corpus_freq = corpus_freqs[key]
        # l = doc_total - doc_freq
        # m = corpus_total - corpus_freqs[key]
        # (j+k)*(j+l)/j+k+l+m; после взаимного уничтожения слагаемых:
        exp_freqs[key] = (doc_freq + corpus_freq) * doc_total / (doc_total + corpus_total)
    return exp_freqs


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
    if not check(expected, dict, exp_cont=str, exp_val=float, not_empty=True) or \
            not check(observed, dict, exp_cont=str, exp_val=int, not_empty=True):
        return None
    chi_values = {}
    for key, value in observed.items():
        chi_values[key] = (value - expected[key]) ** 2 / expected[key]
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
    if not check(chi_values, dict, exp_cont=str, not_empty=True) or \
            not check(alpha, float) or alpha not in criterion.keys():
        return None
    significant_words = {}
    for key, value in chi_values.items():
        if value > criterion[alpha]:
            significant_words[key] = value
    return significant_words
