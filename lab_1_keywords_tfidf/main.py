"""
Lab 1
Extract keywords based on frequency related metrics
"""
# fix gettopn
from typing import Optional, Union
import math


def check_input_type(check_arg, check_type, check_token=None, check_value=None, check_if_not_empty=False):
    if not isinstance(check_arg, check_type):
        return False
    if not check_arg and check_if_not_empty:
        return False
    if check_type == list:
        for i in check_arg:
            if not isinstance(i, check_token):
                return False
            else:
                return True
    if check_type == dict:
        for i in check_arg:
            if not isinstance(i, check_value):
                return False
            else:
                return True


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens
    Parameters:
    text (str): Original text
    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation
    In case of corrupt input arguments, None is returned
    """
    if not check_input_type(text, str):
        return None
    else:
        text_small = text.lower().strip()
        tokens_str = ''
        for i in text_small:
            if i.isalnum() is True or i == ' ' or i == '\n':
                tokens_str += i
            else:
                continue
        tokens = tokens_str.split()
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
    if not check_input_type(tokens, list, str, check_if_not_empty=True) or \
            not check_input_type(stop_words, list, str, check_if_not_empty=True):
        return None
    else:
        no_stop_words = []
        for x in tokens:
            if x not in stop_words:
                no_stop_words.append(x)
            else:
                continue
        return no_stop_words


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence
    Parameters:
    tokens (List[str]): Token sequence to count frequencies for
    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary
    In case of corrupt input arguments, None is returned
    """
    if not check_input_type(tokens, list, str, check_if_not_empty=True):
        return None
    else:
        tokens_freqs = {}
        for i in tokens:
            tokens_freqs[i] = tokens.count(i)
        return tokens_freqs


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
    if not check_input_type(frequencies, dict, str, Union[int, float], check_if_not_empty=True) \
            or not check_input_type(top, int):
        return None
    else:
        sorted_tokens_freqs = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
        if len(frequencies) > top:
            top_freqs_words = list(sorted_tokens_freqs.keys())[:top]
        else:
            top_freqs_words = list(sorted_tokens_freqs.keys())
        return top_freqs_words


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
    if not check_input_type(frequencies, dict, str, int, check_if_not_empty=True):
        return None
    else:
        term_freq = {}
        for k, v in frequencies.items():
            new_v = v / sum(frequencies.values())
            term_freq[k] = new_v
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
    if not check_input_type(term_freq, dict, str, float, check_if_not_empty=True) \
            or not check_input_type(idf, dict, str, float, check_if_not_empty=True):
        return None
    else:
        for k in term_freq:
            if k in idf:
                for k in idf.keys():
                    tf_idf_v = idf[k] * term_freq[k]
                    term_freq.update({k: tf_idf_v})
            else:
                tf_idf_v = math.log(47/1)
                term_freq.update({k: tf_idf_v})
                return term_freq


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
