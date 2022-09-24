"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union

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
        for punc in string.punctuation:
            text = text.replace(punc, '')
        clean_words = text.lower().split()
        return clean_words
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
    if (isinstance(tokens, list) and tokens and all(isinstance(token, str) for token in tokens)
            and isinstance(stop_words, list)):
        no_stop_words = [token for token in tokens if token not in stop_words]
        return no_stop_words
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
    if (isinstance(tokens, list) and tokens
            and all(isinstance(token, str) for token in tokens)):
        freq_dict = {}
        for token in tokens:
            if token not in freq_dict:
                freq_dict[token] = 1
            else:
                freq_dict[token] += 1
        return freq_dict
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
    if (isinstance(frequencies, dict) and frequencies
            and all(isinstance(token, str) for token in frequencies.keys())
            and all(isinstance(freq, (int, float)) for freq in frequencies.values())
            and isinstance(top, int) and top is not (True or False) and top > 0):

        sorted_words = [token for token, freq in sorted(frequencies.items(), key=lambda token: token[1], reverse=True)]
        top_words = sorted_words[:top]
        return top_words
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
    if (isinstance(frequencies, dict) and frequencies
            and all(isinstance(token, str) for token in frequencies.keys())
            and all(isinstance(freq, int) for freq in frequencies.values())):

        words_num = sum(frequencies.values())
        tf_dict = {token: (freq / words_num) for token, freq in frequencies.items()}
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
    if (isinstance(term_freq, dict) and term_freq
            and all(isinstance(token, str) for token in term_freq.keys())
            and all(isinstance(tf, float) for tf in term_freq.values())
            and isinstance(idf, dict) and all(isinstance(token, str) for token in idf.keys())
            and all(isinstance(idf_val, float) for idf_val in idf.values())):

        tfidf_dict = {}
        for token in term_freq:
            if token not in idf.keys():
                idf[token] = math.log(47 / 1)
            tfidf_dict[token] = term_freq[token] * idf[token]
        return tfidf_dict
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
    if (isinstance(doc_freqs, dict) and doc_freqs
            and all(isinstance(token, str) for token in doc_freqs.keys())
            and all(isinstance(freq, int) for freq in doc_freqs.values())
            and isinstance(corpus_freqs, dict)
            and all(isinstance(token, str) for token in corpus_freqs.keys())
            and all(isinstance(freq, int) for freq in corpus_freqs.values())):

        expected_freq_dict = {}

        for token, freq in doc_freqs.items():

            if token not in corpus_freqs.keys():
                corpus_freqs[token] = 0

            collection_freq = corpus_freqs[token]
            partial_doc_words_sum = sum(doc_freqs.values()) - freq
            partial_collection_words_sum = sum(corpus_freqs.values()) - collection_freq

            expected = (((freq + collection_freq) * (freq + partial_doc_words_sum)) /
                        (freq + collection_freq + partial_doc_words_sum + partial_collection_words_sum))

            expected_freq_dict[token] = expected
        return expected_freq_dict
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
    if (isinstance(expected, dict) and expected
            and all(isinstance(token, str) for token in expected.keys())
            and all(isinstance(freq, float) for freq in expected.values())
            and isinstance(observed, dict) and observed != {}
            and all(isinstance(token, str) for token in observed.keys())
            and all(isinstance(freq, int) for freq in observed.values())):

        chi_val_dict = {}
        for token, freq in observed.items():
            chi = (((freq - expected[token]) ** 2) / (expected[token]))
            chi_val_dict[token] = chi
        return chi_val_dict
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
    if (isinstance(chi_values, dict) and chi_values
        and all(isinstance(token, str) for token in chi_values.keys())
        and all(isinstance(chi_val, float) for chi_val in chi_values.values())
        and alpha in [0.05, 0.01, 0.001]):

        criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
        significant_chi_words = {}

        for token, chi_val in chi_values.items():
            if chi_val >= criterion[alpha]:
                significant_chi_words[token] = chi_val
        return significant_chi_words
    return None
