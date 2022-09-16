"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from math import log


def correct_type(d: Union[dict, list], type1: list) -> bool:
    if type(d) == dict:
        for key, value in d.items():
            if type(key) != type1[0] or type(value) != type1[1]:
                return False
            continue
        return True
    elif type(d) == list and d != []:
        for i in d:
            if type(i) != type1[0]:
                return False
            continue
        return True
    else:
        return False


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    new_text = ""
    if type(text) == str:
        for symbol in text.lower():
            if symbol == " " or symbol.isalpha() or symbol == "\n":
                new_text += symbol
        split_words = new_text.split()
        return split_words


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
    not_stop_words = []
    if correct_type(tokens, [str]) is True and correct_type(stop_words, [str]) is True:
        for word in tokens:
            if word not in stop_words:
                not_stop_words.append(word)
        return not_stop_words


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    freq_dict = {}
    if correct_type(tokens, [str]) is True:
        for token in tokens:
            if token in freq_dict:
                freq_dict[token] += 1
            else:
                freq_dict[token] = 1
        return freq_dict


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
    top_words = []
    if correct_type(frequencies, [str, int]) is True or correct_type(frequencies, [str, float]) is True:
        sorted_freq = sorted(frequencies.values(), reverse=True)
        for i in sorted_freq:
            for key, value in frequencies.items():
                if value == i and len(top_words) < top:
                    top_words.append(key)
        return top_words


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
    tf_dict = {}
    if correct_type(frequencies, [str, int]) is True:
        for key, value in frequencies.items():
            tf_dict[key] = value / sum(frequencies.values())
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
    tfidf = {}
    if correct_type(term_freq, [str, float]) is True and correct_type(idf, [str, float]) is True:
        for key, value in term_freq.items():
            try:
                tfidf[key] = term_freq[key] * idf[key]
            except KeyError:
                idf[key] = log(47 / (0 + 1)) * term_freq[key]
        return tfidf


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
    expected_dict = {}
    if correct_type(doc_freqs, [str, int]) is True and correct_type(corpus_freqs, [str, int]) is True:
        for key, value in doc_freqs.items():
            l = sum(doc_freqs.values()) - value
            m = sum(corpus_freqs.values()) - value
            k = corpus_freqs.get(key, 0)
            expected_dict[key] = ((value + k) * (value + l)) / (value + k + l + m)
        return expected_dict


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
    x2_dict = {}
    if correct_type(expected, [str, float]) is True and correct_type(observed, [str, int]) is True:
        for key, value in expected.items():
            x2_dict[key] = (observed.get(key) - value) ** 2 / value
        return x2_dict


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
    significant_words_dict = {}
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if correct_type(chi_values, [str, float]) is True and type(alpha) == float:
        for key, value in chi_values.items():
            if value > criterion[alpha]:
                significant_words_dict[key] = value
        return significant_words_dict
