"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any
from math import log


def check_list(user_input: Any, elements_type: type, can_be_empty: bool) -> bool:
    """
    Checks weather object is list
    that contains objects of certain type
    """
    if not isinstance(user_input, list):
        return False
    if not user_input and can_be_empty is False:
        return False
    for element in user_input:
        if not isinstance(element, elements_type):
            return False
    return True


def check_dict(user_input: dict, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    """
    Checks weather object is dictionary
    hat has keys and values of certain type
    """
    if not isinstance(user_input, dict):
        return False
    if not user_input and can_be_empty is False:
        return False
    for key, value in user_input.items():
        if not (isinstance(key, key_type) and isinstance(value, value_type)):
            return False
    return True


def check_positive_int(user_input: Any) -> bool:
    """
    Checks weather object is int (not bool)
    """
    if not isinstance(user_input, int):
        return False
    if isinstance(user_input, bool):
        return False
    if user_input <= 0:
        return False
    return True


def check_float(user_input: Any) -> bool:
    """
    Checks weather object is float
    """
    if not isinstance(user_input, float):
        return False
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
    if not isinstance(text, str):
        return None
    punctuation = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'''
    my_text = ''
    for char in text.lower().replace('\n', ' '):
        if char not in punctuation:
            my_text += char
    return my_text.split()


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
    my_tokens = []
    if not (check_list(tokens, str, False) and check_list(stop_words, str, True)):
        return None
    for token in tokens:
        if token not in stop_words:
            my_tokens.append(token)
    return my_tokens


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not check_list(tokens, str, False):
        return None
    if tokens:
        my_dict = {token: tokens.count(token) for token in tokens}
    return my_dict


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
    checking_dict = check_dict(frequencies, str, int, False) or check_dict(frequencies, str, float, False)
    if not (checking_dict and check_positive_int(top)):
        return None
    return sorted(frequencies.keys(), key=lambda key: frequencies[key], reverse=True)[:top]