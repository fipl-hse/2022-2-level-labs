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
    text = text.lower()
    for bad_symbol in '.,:-!?;%<>&*@#()':
        text = text.replace(bad_symbol, '')
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
    if not isinstance(tokens, list) or tokens == []:
        return None
    frequency_dict = {}
    for token in tokens:
        if not isinstance(token, str):
            return None
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

    def sort_key(word_and_frequency: tuple[str, Union[int, float]]):
        return word_and_frequency[1]

    if not isinstance(frequencies, dict) or frequencies == {} \
            or not isinstance(top, int) or isinstance(top, bool) or top <= 0:
        return None
    top_list = []
    for item in frequencies.items():
        if not isinstance(item[0], str) or not (isinstance(item[1], int) or isinstance(item[1], float)):
            return None
        top_list.append(item)
    top_list.sort(reverse=True, key=sort_key)
    top_list = top_list[:top]
    for index in range(len(top_list)):
        top_list[index] = top_list[index][0]
    return top_list


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
    if not isinstance(frequencies, dict):
        return None
    total_words = 0
    tf_dict = {}
    for token, frequency in frequencies.items():
        if not isinstance(token, str):
            return None
        total_words += frequency
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
    if not isinstance(term_freq, dict) or not isinstance(idf, dict) or term_freq == {}:
        return None
    tfidf_dict = {}
    for key, value in term_freq.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None
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
    if not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict) or doc_freqs == {}:
        return None
    doc_total = 0
    for key, value in doc_freqs.items():
        if not isinstance(key, str):
            return None
        doc_total += value
    corpus_total = 0
    for key, value in corpus_freqs.items():
        if not isinstance(key, str):
            return None
        corpus_total += value
    exp_freqs = {}
    for key, doc_freq in doc_freqs.items():
        # j = doc_freq
        # k = corpus_freqs[key] (0 if does not appear corpus)
        corpus_freq = 0
        if key in corpus_freqs.keys():
            corpus_freq = corpus_freqs[key]
        # l = doc_total - doc_freq
        # m = corpus_total - corpus_freqs[key]
        # (j+k)*(j+l)/j+k+l+m
        # после взаимного уничтожения слагаемых:
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
    if not isinstance(expected, dict) or not isinstance(observed, dict) or expected == {} or observed == {}:
        return None
    chi_values = {}
    for key, value in expected.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None
    for key, value in observed.items():
        if not isinstance(key, str) or not isinstance(value, int):
            return None
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
    CRITERION = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if not isinstance(chi_values, dict) or chi_values == {} or not isinstance(alpha, float) \
            or not alpha in CRITERION.keys():
        return None
    significant_words = {}
    for key, value in chi_values.items():
        if not isinstance(key, str):
            return None
        if value > CRITERION[alpha]:
            significant_words[key] = value
    return significant_words
