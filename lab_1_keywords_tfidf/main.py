"""
Lab 1
Extract keywords based on frequency related metrics
"""
import math
from typing import Optional, Union
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
        lowered_text = text.lower()
        for punctuation_mark in string.punctuation:
            if punctuation_mark in lowered_text:
                lowered_text = lowered_text.replace(punctuation_mark, '')
        cleaned_text = lowered_text.strip().split()
        return cleaned_text
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
    if not isinstance(tokens, (str, list)) or not isinstance(stop_words, (list, str)):
        return None
    index = 0
    while index < len(tokens):
        for stop_word in stop_words:
            if tokens[index] == stop_word:
                tokens.remove(stop_word)
                index -= 1
                break
        index += 1
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
    if isinstance(tokens, list) and len(tokens) != 0:
        frequency_dict = {}
        for token in tokens:
            if isinstance(token, str):
                if token in frequency_dict.keys():
                    frequency_dict[token] = 1 + frequency_dict[token]
                else:
                    frequency_dict[token] = 1
            else:
                return None
        return frequency_dict
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
    if not isinstance(frequencies, dict) or frequencies == {} or not isinstance(top, int) or isinstance(top, bool) or top <= 0:
        return None
    frequencies = sorted(frequencies.items(), reverse=True, key=lambda item: item[1])
    if len(frequencies) < top:
        top_list = []
        for word in frequencies:
            top_list.append(word[0])
        return top_list
    elif len(frequencies) >= top:
        top_list = []
        for word in frequencies:
            top_list.append(word[0])
        top_n_words = top_list[:top]
        return top_n_words




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
    all_words = 0
    new_dict = {}
    if not isinstance(frequencies, dict):
        return None
    for key, value in frequencies.items():
        if not isinstance(key, str):
            return None
        all_words += value
    for key in frequencies.keys():
        new_dict[key] = frequencies[key] / all_words
    return new_dict




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
    if term_freq == {} or not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None
    final_dict = {}
    for key_freq, value_freq in term_freq.items():
        if not isinstance(key_freq, str) or not isinstance(value_freq, float):
            return None
        if key_freq in idf:
            final_dict[key_freq] = value_freq * idf[key_freq]
        else:
            final_dict[key_freq] = value_freq * math.log(47)
    return final_dict




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
