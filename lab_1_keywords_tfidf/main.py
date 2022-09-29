"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from operator import itemgetter
from math import log
from string import punctuation


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
    for i in punctuation:
        text = text.replace(i, "")
        text = text.lower().strip()
    tokens = text.split()
    print(tokens)
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
    if not isinstance(tokens, (str, list)) or not isinstance(stop_words, (list, str)):
        return None
    filtered_text = []
    for i in tokens:
        if i not in stop_words:
            filtered_text.append(i)
    print(filtered_text)
    return filtered_text


def calculate_frequencies(filtered_text: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(filtered_text, list)) or not filtered_text:
        return None
    for i in filtered_text:
        if not isinstance(i, str):
            return None
    dictionary = {}
    for i in filtered_text:
        dictionary[i] = dictionary.get(i, 0) + 1
    print(dictionary)
    return dictionary


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

    if not isinstance(frequencies, dict) or not isinstance(top, int) or isinstance(top, bool) or not frequencies or \
            top <= 0:
        return None
    for key in frequencies.keys():
        if not isinstance(key, str):
            return None
    for value in frequencies.values():
        if not isinstance(value, int) and not isinstance(value, float):
            return None
        sorted_dictionary = [token for token, word in
                             sorted(frequencies.items(), key=itemgetter(1), reverse=True)[:top]]
        return sorted_dictionary


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
    if not (isinstance(frequencies, dict) and frequencies
            and all(isinstance(token, str) for token in frequencies.keys())
            and all(isinstance(value, int) for value in frequencies.values())):
        return None
    dictionary_tf = {}
    number = sum(frequencies.values())
    for a, b in frequencies.items():
        if not isinstance(a, str) and not isinstance(b, int):
            return None
        dictionary_tf[a] = b / number
    print(dictionary_tf)
    return dictionary_tf


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
    if not (isinstance(term_freq, dict) and term_freq
            and isinstance(idf, dict) and all(isinstance(token, str) for token in idf.keys())
            and all(isinstance(token, str) for token in term_freq.keys())
            and all(isinstance(tf_meaning, float) for tf_meaning in term_freq.values())
            and all(isinstance(idf_value, float) for idf_value in idf.values())):
        return None
    dict_tf_idf = {}
    for token in term_freq:
        if token not in idf.keys():
            idf[token] = log(47 / (0 + 1))
        dict_tf_idf[token] = term_freq[token] * idf[token]
    print(dict_tf_idf)
    return dict_tf_idf


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
    if not doc_freqs or not isinstance(corpus_freqs, dict) or not isinstance(doc_freqs, dict):
        return None
    for key, value in corpus_freqs.items():
        if not isinstance(key, str) or not isinstance(value, int):
            return None
    for key, value in doc_freqs.items():
        if not isinstance(key, str) or not isinstance(value, int):
            return None
    all_words_d = sum(doc_freqs.values())
    all_words_c = sum(corpus_freqs.values())
    expected_frequency = {}
    for key in doc_freqs:
        j = doc_freqs.get(key, 0)
        l = all_words_d - doc_freqs.get(key, 0)
        k_corp = corpus_freqs.get(key, 0)
        m = all_words_c - corpus_freqs.get(key, 0)
        expected_frequency[key] = ((j + k_corp) * (j + l)) / (j + k_corp + l + m)
    print(expected_frequency)
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
