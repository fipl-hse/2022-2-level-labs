"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union

# import json
# from pathlib import Path

import math
import string


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens
    Parameters:
    text (str): Original text
    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation
    In case of corrupt input arguments, None is returned
    """
    if isinstance(text, str):
        for p in string.punctuation:
            text = text.replace(p, '')
        clean_words = text.lower().strip().split()
        return clean_words
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
    if (isinstance(tokens, list) and tokens != [] and all(isinstance(t, str) for t in tokens)
            and isinstance(stop_words, list)):
        no_stop_words = [t for t in tokens if t not in stop_words]
        return no_stop_words
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
    if (isinstance(tokens, list) and tokens != []
            and all(isinstance(t, str) for t in tokens)):
        freq_dict = {}
        for t in tokens:
            if t not in freq_dict:
                freq_dict[t] = 1
            else:
                freq_dict[t] += 1
        return freq_dict
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
    if (isinstance(frequencies, dict) and frequencies != {}
            #and all(isinstance(k, str) for k in frequencies.keys())
            #and all(isinstance(v, int or float) for v in frequencies.values())
            and isinstance(top, int) and top is not (True or False) and top > 0):

        sorted_freq_dict = {k: v for k, v in sorted(frequencies.items(), key=lambda k: k[1], reverse=True)}
        sorted_words = list(sorted_freq_dict.keys())
        top_words = sorted_words if top > len(frequencies) else sorted_words[:top]
        return top_words

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
    if (isinstance(frequencies, dict) and frequencies != {}
            and all(isinstance(k, str) for k in frequencies.keys())
            and all(isinstance(v, int or float) for v in frequencies.values())):
        words_num = sum(frequencies.values())
        tf_dict = {w: (f / words_num) for w, f in frequencies.items()}
        return tf_dict
    else:
        None


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
    if (isinstance(term_freq, dict) and term_freq != {} and all(isinstance(w, str) for w in term_freq.keys())
            and all(isinstance(f, float) for f in term_freq.values())
            and isinstance(idf, dict) and all(isinstance(w, str) for w in idf.keys())
            and all(isinstance(f, float) for f in idf.values())):

        tfidf_dict = {}
        for w in term_freq:
            if w not in idf.keys():
                idf[w] = math.log(47 / 1)
            tfidf_dict[w] = term_freq[w] * idf[w]
        return tfidf_dict
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
    if (isinstance(doc_freqs, dict) and doc_freqs != {} and all(isinstance(k, str) for k in doc_freqs.keys())
            and all(isinstance(v, int) for v in doc_freqs.values())
            and isinstance(corpus_freqs, dict) and all(isinstance(k, str) for k in corpus_freqs.keys())
            and all(isinstance(v, int) for v in corpus_freqs.values())):

        expected_dict = {}
        doc_words_sum = sum(doc_freqs.values())
        collection_words_sum = sum(corpus_freqs.values())
        for w, f in doc_freqs.items():
            expected = (((f + corpus_freqs[w]) * (f + doc_words_sum - f)) /
                        (f + corpus_freqs[w] + doc_words_sum - f + collection_words_sum - corpus_freqs[w]))
            expected_dict[w] = expected
        return expected_dict
    else:
        return


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
    if (isinstance(expected, dict) and expected != {}
            and all(isinstance(k, str) for k in expected.keys())
            and all(isinstance(v, float) for v in expected.values())
            and isinstance(observed, dict) and observed != {}
            and all(isinstance(k, str) for k in observed.keys())
           and all(isinstance(v, int) for v in observed.values())):

        chi_dict = {}
        for w, f in observed.items():
            chi = (((f - expected[w]) ** 2) / (expected[w]))
            chi_dict[w] = chi
        return chi_dict
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
    pass
