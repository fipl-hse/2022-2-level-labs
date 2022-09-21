"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from operator import itemgetter

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
        'The isinstance() function returns True if the specified object is of the specified type, otherwise False.'
        return None
    'if var text is not a string - return None'
    text = text.lower()
    'make entire var text lowercase'
    punctuation_symbols = '.,:-!?;%<>&*@#()' 'creating a var that has all the punctuation marks'
    for i in punctuation_symbols:
        'go through each punctuation symbol and replace it in the text by noting'
        text = text.replace(i, '')
    text_split = text.split()
    'splits text by spaces'
    return text_split
    pass

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
    if not (isinstance(tokens, list) and isinstance(stop_words, list)):
        'The isinstance() function returns True if the specified object is of the specified type, otherwise False.'
        return None
    stop_words_removed = [x for x in tokens if x not in stop_words]
    return stop_words_removed
    pass


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
        'The isinstance() function returns True if the specified object is of the specified type, otherwise False.'
        return None
    Dict = {}
    for i in tokens:
        if i in Dict.keys():
            Dict[i] = 1 + Dict[i]
        else:
            Dict[i] = 1
    return (Dict)
    pass


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
    if not (isinstance(frequencies, dict), isinstance(top, str)):
        'The isinstance() function returns True if the specified object is of the specified type, otherwise False.'
        return None
    "If you want to conserve all the information from a dictionary when sorting it," \
    " the typical first step is to call the .items() method on the dictionary. " \
    "Calling .items() on the dictionary will provide an iterable of tuples representing the key-value pairs:" \
    "Tuple(кортеж) is a Python type that is an ordered collection of objects"
    sorted_frequencies = dict(sorted(frequencies.items()))
    'operator is a built-in module providing a set of convenient operators. operator.itemgetter(n) ' \
    'constructs a callable that assumes an iterable object (e.g. list, tuple, set) as input, ' \
    'and fetches the n-th element out of it.'
    List = sorted_frequencies[:top]
    pass


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
