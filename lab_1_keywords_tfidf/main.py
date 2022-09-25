"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import math
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
    text = text.lower()
    for i in text:
        if i in punctuation:
            text = text.replace(i, '')
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
    for i in stop_words:
        if not isinstance(i, str):
            return None
    for i in tokens:
        if not isinstance(i, str):
            return None
    tokens_new = []
    for i in tokens:
        if i not in stop_words:
            tokens_new.append(i)
    return tokens_new


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
    for i in tokens:
        if not isinstance(i, str):
            return None
    new_dict = {}
    for i in tokens:
        if i not in new_dict:
            new_dict[i] = 1
        else:
            new_dict[i] += 1
    return new_dict


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
    if not frequencies or not top:
        return None
    if not type(top).__name__ == 'int' or not isinstance(frequencies, dict):
        return None
    for i in frequencies.keys():
        if not isinstance(i, str):
            return None
    for i in frequencies.values():
        if not isinstance(i, int) and not isinstance(i, float):
            return None
    sorted_keys = sorted(frequencies, key=frequencies.get, reverse=True)
    sorted_top = sorted_keys[:top]
    return sorted_top


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
    if not frequencies or not isinstance(frequencies, dict):
        return None
    for i in frequencies.keys():
        if not isinstance(i, str):
            return None
    for i in frequencies.values():
        if not isinstance(i, int):
            return None
    summa = 0  # количество слов
    freq_new = {}
    for val in frequencies.values():
        summa += val
    for key in frequencies.keys():
        freq_new[key] = frequencies[key] / summa
    return freq_new


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
    if not term_freq:
        return None
    if not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None
    for key in term_freq.keys():
        for value in term_freq.values():
            if not isinstance(key, str) or not isinstance(value, float):
                return None
    for key in idf.keys():
        for value in idf.values():
            if not isinstance(key, str) or not isinstance(value, float):
                return None
    new = {}
    for term_key in term_freq.keys():
        if term_key not in idf:
            new[term_key] = term_freq[term_key] * math.log(47/1)
        else:
            new[term_key] = term_freq[term_key] * idf[term_key]
    return new


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
    if not doc_freqs or not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict):
        return None
    for i in doc_freqs.keys():
        for j in doc_freqs.values():
            if not isinstance(i, str) or not isinstance(j, int):
                return None
    for i in corpus_freqs.keys():
        for j in corpus_freqs.values():
            if not isinstance(i, str) or not isinstance(j, int):
                return None
    expected_freq = {}
    all_doc = sum(doc_freqs.values())
    all_corpus = sum(corpus_freqs.values())
    for e in doc_freqs.keys():
        t_in_d = doc_freqs[e]
        not_in_d = all_doc - doc_freqs[e]
        if e in corpus_freqs.keys():
            t_in_all = corpus_freqs[e]
            not_in_all = all_corpus - corpus_freqs[e]
        else:
            t_in_all = 0
            not_in_all = all_corpus
        expected_freq[e] = ((t_in_d+t_in_all)*(t_in_d+not_in_d)) / (t_in_d+t_in_all+not_in_d+not_in_all)
    return expected_freq


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
    if not expected or not observed:
        return None
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        return None
    for i in expected.keys():
        for j in expected.values():
            if not isinstance(i, str) or not isinstance(j, float):
                return None
    for i in observed.keys():
        for j in observed.values():
            if not isinstance(i, str) or not isinstance(j, int):
                return None
    xi_val = {}
    for keys in expected.keys():
        obs_keys = observed[keys]
        exp_keys = expected[keys]
        xi_val[keys] = ((obs_keys - exp_keys) ** 2) / exp_keys
    return xi_val


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
    if not chi_values or not alpha:
        return None
    if not isinstance(chi_values, dict) or not isinstance(alpha, float):
        return None
    for i in chi_values.keys():
        for j in chi_values.values():
            if not isinstance(i, str) or not isinstance(j, float):
                return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None
    signific = {}
    for key, val in chi_values.items():
        if val > criterion[alpha]:
            signific[key] = val
    return signific
