"""
Lab 1
Extract keywords based on frequency related metrics
"""
import math
from typing import Optional, Union
import string


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    if isinstance(text, str) == True:
        text = text.lower()
        for a in string.punctuation:
            if a in text:
                text = text.replace(a, '')
        text = text.strip().split()
        return text
    else:
        return None

    """
    Removes punctuation, casts to lowercase, splits into tokens
    Parameters:
    text (str): Original text
    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation
    In case of corrupt input arguments, None is returned
    """


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    if not isinstance(tokens, (str, list)) or not isinstance(stop_words, (list, str)):
        return None
    b = 0
    while b < len(tokens):
        for c in stop_words:
            if tokens[b] == c:
                tokens.remove(c)
                b -= 1
                break
        b += 1
    return tokens

    """
    Excludes stop words from the token sequence
    Parameters:
    tokens (List[str]): Original token sequence
    stop_words (List[str]: Tokens to exclude
    Returns:
    List[str]: Token sequence that does not include stop words
    In case of corrupt input arguments, None is returned
    """


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    if isinstance(tokens, list) == True and len(tokens) != 0:
        d = {}
        for i in tokens:
            if type(i) == str:
                if i in d.keys():
                    d[i] = 1 + d[i]
                else:
                    d[i] = 1
            else:
                return None
        return d
    else:
        return None

    """
    Composes a frequency dictionary from the token sequence
    Parameters:
    tokens (List[str]): Token sequence to count frequencies for
    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary
    In case of corrupt input arguments, None is returned
    """


def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
    if not isinstance(frequencies, dict) or frequencies == {} or not isinstance(top, int) or isinstance(top, bool) or top <= 0:
        return None
    frequencies = sorted(frequencies.items(), reverse=True, key=lambda item: item[1])
    if len(frequencies) < top:
        list1 = []
        for i in frequencies:
            list1.append(i[0])
        return list1
    elif len(frequencies) >= top:
        list1 = []
        for i in frequencies:
            list1.append(i[0])
        d = list1[:top]
        return d

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


def calculate_tf(frequencies: dict[str, int]) -> Optional[dict[str, float]]:
    all_words = 0
    new_dict = {}
    if not isinstance(frequencies, dict):
        return None
    for key, value in frequencies.items():
        if not isinstance(key, str):
            return None
        all_words += value
    for key in frequencies.keys():
        new_dict[key] = frequencies[key] / all_words
    return new_dict

    """
    Calculates Term Frequency score for each word in a token sequence
    based on the raw frequency
    Parameters:
    frequencies (Dict): Raw number of occurrences for each of the tokens
    Returns:
    dict: A dictionary with tokens and corresponding term frequency score
    In case of corrupt input arguments, None is returned
    """


def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> Optional[dict[str, float]]:
    if term_freq == {} or not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None
    final_dict = {}
    for key_freq, value_freq in term_freq.items():
        if not isinstance(key_freq, str) or not isinstance(value_freq, float):
            return None
        if key_freq in idf:
            final_dict[key_freq] = value_freq * idf[key_freq]
        else:
            final_dict[key_freq] = value_freq * math.log(47)
    return final_dict

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
