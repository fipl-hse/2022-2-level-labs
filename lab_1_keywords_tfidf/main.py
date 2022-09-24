"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from string import punctuation
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
    if isinstance(text, str):
        clean_text = ''
        text = text.lower().strip().replace('\n', ' ')
        for token in text:
            if token not in punctuation:
                clean_text += token
        clean_text = clean_text.split()
        return clean_text
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
    if isinstance(tokens,list) and all(isinstance(token, str) for token in tokens) and isinstance(
        stop_words,list) and all(isinstance(token, str) for token in stop_words):
        clean_tokens = []
        for token in tokens:
            if token not in stop_words:
                clean_tokens.append(token)
        return clean_tokens
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
    if isinstance(tokens, list) and all(isinstance(token, str) for token in tokens):
        fr_dict = {}
        for token in tokens:
            fr_dict[token] = tokens.count(token)
        return fr_dict
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
    if isinstance(frequencies, dict) and all(isinstance(token, str) for token in frequencies.keys()) and isinstance(
            top, int):
        for value in frequencies.values():
            if not isinstance(value, int) and not isinstance(value, float):
                return None
        top_n = sorted(frequencies.keys(), key=lambda token: frequencies.get(token), reverse=True)[:top]
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
    if isinstance(frequencies, dict) and all(isinstance(token, str) for token in frequencies.keys()) and all(
            isinstance(values, int) for values in frequencies.values()):
        tf_dict = {}
        for token, frequence in frequencies.items():
            tf_dict[token] = frequence / sum(frequencies.values())
        return tf_dict
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
    if isinstance(term_freq, dict) and all(isinstance(token, str) for token in term_freq.keys()) and all(
            isinstance(value, float) for value in term_freq.values()) and isinstance(idf, dict) and all(
            isinstance(token, str) for token in idf.keys()) and all(isinstance(value, float) for value in idf.values()):
        tfidf_dict = {}
        for token in term_freq.keys():
            if token in idf.keys():
                tfidf_dict[token] = term_freq.get(token) * idf.get(token)
            else:
                tfidf_dict[token] = term_freq.get(token) * log(47 / (0 + 1))
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
    if isinstance(doc_freqs, dict) and all(isinstance(token, str) for token in doc_freqs.keys()) and all(
            isinstance(value, int) for value in doc_freqs.values()) and isinstance(corpus_freqs, dict) and all(
            isinstance(token, str) for token in corpus_freqs.keys()) and all(
            isinstance(value, int) for value in corpus_freqs.values()):
        exp_freq_dict = {}
        for token, value in doc_freqs.items():
            value_corpus = corpus_freqs.get(token)
            all_doc = sum(doc_freqs.values()) - value
            all_corpus = sum(corpus_freqs.values()) - value_corpus
            exp_freq_dict[token] = (value + value_corpus) * (value + all_doc) / (
                    value + value_corpus + all_doc + all_corpus)
        return exp_freq_dict
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
    if isinstance(expected, dict) and all(isinstance(token, str) for token in expected.keys()) and all(
            isinstance(value, float) for value in expected.values()) and isinstance(observed, dict) and all(
            isinstance(token, str) for token in observed.keys()) and all(
            isinstance(value, int) for value in observed.values()):
        chi_dict = {}
        for token, value in expected.items():
            value_observed = observed.get(token)
            chi_dict[token] = (value_observed + value)**2 / value
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
    if isinstance(chi_values, dict) and all(isinstance(token, str) for token in chi_values.keys()) and all(
            isinstance(value, float) for value in chi_values.values()) and isinstance(alpha, float):
        CRITERION = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
        if alpha in CRITERION.keys():
            alpha_value = CRITERION.get(alpha)
        else:
            return None
        significant_words = {}
        for token, value in chi_values.items():
            if value >= alpha_value:
                significant_words[token] = value
        return significant_words
    else:
        return None
