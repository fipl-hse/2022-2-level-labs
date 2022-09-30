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
    target_text = text.lower().strip()
    symbols = """!()-[]{};?@#$%:'",.^&;*_><"""
    for symbol in target_text:
        if symbol in symbols:
            target_text = target_text.replace(symbol, '')
    tokens = target_text.split()
    for word in tokens:
        if word == "' '":
            tokens.remove(word)
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
    if not isinstance(tokens, list) or not isinstance(stop_words, list):
        return None
    if not (isinstance(word1, str) for word1 in stop_words) or not (isinstance(word2, str) for word2 in tokens):
        return None
    new_tokens = []
    for word3 in tokens:
        if word3 not in stop_words:
            new_tokens += [word3]
    tokens = new_tokens
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
    if not isinstance(tokens, list):
        return None
    frequencies = {}
    for word in tokens:
        if not isinstance(word, str) or word == "":
            return None
        frequencies[word] = tokens.count(word)
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
    if not ((isinstance(top, int) and top > 0 and isinstance(top, bool) is not True and isinstance(frequencies, dict)
             and frequencies != {})):
        return None
    if not (isinstance(value, (int, float)) for value in frequencies.values()):
        return None
    most_common = sorted(frequencies, key=lambda value: frequencies[value], reverse=True)
    most_common = most_common[:top]
    if most_common is None:
        return None
    return most_common


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
    term_freq = {}
    if not isinstance(frequencies, dict):
        return None
    for key in frequencies.keys():
        if not isinstance(key, str) or key is None:
            return None
    lenght = 0
    for value in frequencies.values():
        if not isinstance(value, int):
            return None
        lenght += value
    for word in frequencies.keys():
        freq = frequencies[word] / lenght
        term_freq[word] = freq
    return term_freq


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

    if not((isinstance(term_freq, dict) and isinstance(idf, dict) and term_freq != {}
            and (all(isinstance(key, str) for key in term_freq.keys())
                 and all(isinstance(value, float) for value in term_freq.values())))):
        return None
    if not((all(isinstance(key, str) for key in idf.keys())
            and all(isinstance(value, float) for value in idf.values()))):
        return None
    tfidf = {}
    if idf == {}:
        for word, word_freq in term_freq.items():
            tfidf[word] = word_freq * math.log(47)
        return tfidf
    for word in idf.keys():
        if word not in idf.keys():
            idf_meaning = math.log(47)
            idf[word] = idf_meaning
            return idf
    for number in idf.values():
        if number == 0:
            max_idf = math.log(47)
            tfidf = {term: term_freq * idf.get(term, max_idf) for term, term_freq in term_freq.items()}
            return tfidf
    return tfidf


def calculate_expected_frequency(doc_freqs: dict[str, int], corpus_freqs: dict[str, int]
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
