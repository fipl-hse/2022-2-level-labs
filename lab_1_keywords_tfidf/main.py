"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from math import log


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
    else:
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
    if isinstance(tokens, list) and all(isinstance(i, str) for i in tokens) and len(
            tokens) != 0 and isinstance(stop_words, list) and all(isinstance(i, str) for i in stop_words):
        res_tokens = []
        for i in tokens:
            if i not in stop_words:
                res_tokens.append(i)
        return res_tokens
    else:
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
    if isinstance(tokens, list) and all(isinstance(i, str) for i in tokens) and len(tokens) != 0:
        frequencies = {}
        for i in tokens:
            if i not in frequencies.keys():
                frequencies[i] = 1
            else:
                frequencies[i] += 1
        return frequencies
    else:
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
    if isinstance(frequencies, dict) and all(isinstance(i, str) for i in frequencies.keys()) and all(
            isinstance(i, Union[int, float]) for i in frequencies.values()) and len(
            frequencies) != 0 and isinstance(top, int):
        sorted_dict = {}
        for i in sorted(frequencies, key=frequencies.get, reverse=True):
            sorted_dict[i] = frequencies[i]
        top_n = list(sorted_dict.keys())[:top]
        return top_n
    else:
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
    if isinstance(frequencies, dict) and all(isinstance(i, str) for i in frequencies.keys()) and all(
            isinstance(i, int) for i in frequencies.values()) and len(frequencies) != 0:
        tf = {}
        nd = len(frequencies)
        for i in frequencies.keys():
            nt = frequencies.get(i)
            token_tf = nt / nd
            tf[i] = token_tf
        return tf
    else:
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
    if isinstance(term_freq, dict) and all(isinstance(i, str) for i in term_freq.keys()) and all(
            isinstance(i, float) for i in term_freq.values()) and len(term_freq) != 0 and isinstance(
            idf, dict) and all(isinstance(i, str) for i in idf.keys()) and all(
            isinstance(i, float) for i in idf.values()) and len(idf) != 0:
        tfidf = {}
        for i in term_freq.keys():
            token_tf = term_freq.get(i)
            if i in idf.keys():
                token_idf = idf.get(i)
            else:
                token_idf = log(47 / (0 + 1))
            token_tfidf = token_tf * token_idf
            tfidf[i] = token_tfidf
        return tfidf
    else:
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
    if isinstance(doc_freqs, dict) and all(isinstance(i, str) for i in doc_freqs.keys()) and all(
            isinstance(i, int) for i in doc_freqs.values()) and len(doc_freqs) != 0 and isinstance(
            corpus_freqs, dict) and all(isinstance(i, str) for i in corpus_freqs.keys()) and all(
            isinstance(i, int) for i in corpus_freqs.values()) and len(corpus_freqs) != 0:
        expected = {}
        for i in doc_freqs.keys():
            j = doc_freqs.get(i)
            if i in corpus_freqs.keys():
                k = corpus_freqs.get(i)
            else:
                k = 0
            l = sum(doc_freqs.values()) - j
            m = sum(corpus_freqs.values()) - k
            expected_token = ((j + k) * (j + l)) / (j + k + l + m)
            expected[i] = expected_token
        return expected
    else:
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
    if isinstance(expected, dict) and all(isinstance(i, str) for i in expected.keys()) and all(
            isinstance(i, float) for i in expected.values()) and len(expected) != 0 and isinstance(
            observed, dict) and all(isinstance(i, str) for i in observed.keys()) and all(
            isinstance(i, int) for i in observed.values()) and len(observed) != 0:
        chi_values = {}
        for i in expected.keys():
            expected_token = expected.get(i)
            observed_token = observed.get(i)
            chi = ((observed_token - expected_token) ** 2) / expected_token
            chi_values[i] = chi
        return chi_values
    else:
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
    if isinstance(chi_values, dict) and all(isinstance(i, str) for i in chi_values.keys()) and all(
            isinstance(i, float) for i in chi_values.values()) and len(chi_values) != 0 and isinstance(
            alpha, float):
        significant_words = {}
        for i in chi_values.keys():
            token_key = chi_values.get(i)
            if token_key >= alpha:
                significant_words[i] = token_key
        return significant_words
    else:
        return None
