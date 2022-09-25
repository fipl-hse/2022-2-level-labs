"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
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
    if isinstance(text, str):
        text = text.lower().strip()
        for symbol in punctuation:
            if symbol in text:
                text = text.replace(symbol, '')
        tokens = text.split()
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
    if isinstance(tokens, list) and isinstance(stop_words, list):
        for stop_word in stop_words:
            if stop_word in tokens:
                while stop_word in tokens:
                    tokens.remove(stop_word)
        return tokens
    else:
        return None


def check_content(massive, type_name):
    if massive and all(isinstance(el, type_name) for el in massive):
        return True
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
    if isinstance(tokens, list) and check_content(tokens, str):
        frequencies = {}
        for token in tokens:
            occurance_number = tokens.count(token)
            frequencies[token] = occurance_number
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
    if not (isinstance(frequencies, dict) and (isinstance(top, int) or isinstance(top, float))):
        return None
    else:
        key_list = list(frequencies.keys())
        value_list = list(frequencies.values())
        if check_content(key_list, str) and (check_content(value_list, int) or check_content(value_list, float)):
            value_list.sort(reverse=True)
            top_list = []

            def get_token(top_list: dict[str], frequencies: dict[str, Union[int, float]], value: int):
                for k, v in frequencies.items():
                    if v == value:
                        if k in top_list:
                            continue
                        else:
                            return k

            def getting_list(length: int):
                for i in range(length):
                    token = get_token(top_list, frequencies, int(value_list[i]))
                    top_list.append(token)
                return top_list
            if len(top_list) != top:
                if top > len(key_list):
                    top_list = getting_list(len(key_list))
                    return top_list
                else:
                    top_list = getting_list(top)
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
    pass


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
    pass


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
