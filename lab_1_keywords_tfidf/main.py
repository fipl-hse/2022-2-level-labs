"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    if type(text) == str:
        clean_text = []
        for element in text:
            if element not in '.,?!;:-()—"\'':
                clean_text.append(element)
        if clean_text[0] == '':
            clean_text.pop(0)
        if clean_text[-1] == '':
            clean_text.pop(-1)
        clean_text = ''.join(clean_text)
        clean_text = clean_text.replace('  ', ' ')
        clean_text = clean_text.lower()
        clean_text = clean_text.split()
        return clean_text
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """

def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    if (type(tokens) == str and all(type(i) == str for i in tokens) == True
            and type(stop_words) == list
            and all(type(i) == str for i in stop_words) == True):
        clean_tokens = []
        for element in tokens:
            if element not in stop_words:
                clean_tokens.append(element)
        return clean_tokens

    """
    Excludes stop words from the token sequence

    Parameters:
    tokens (List[str]): Original token sequence
    stop_words (List[str]: Tokens to exclude

    Returns:
    List[str]: Token sequence that does not include stop words

    In case of corrupt input arguments, None is returned
    """

def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    if type(tokens) == list and all(type(i) == str for i in tokens) == True and tokens != []:
        words_and_frequencies = {}
        for element in tokens:
            if element not in words_and_frequencies:
                count_element = tokens.count(element)
                words_and_frequencies[element] = count_element
        return words_and_frequencies

    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """

def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
    if type(frequencies) == dict and all(type(i) == str for i in frequencies.keys()) == True and (all(
        type(c) == int for c in frequencies.values()) == True or all(
        type(c) == float for c in frequencies.values())) == True and type(top) == int:
        reversed_dict = dict(sorted(list(frequencies.items()), key=lambda c: c[1], reverse=True))
        top_n = []
        for i in range(top):
            top_n.append(list(reversed_dict.keys())[i])
        return top_n

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

def calculate_tf(frequencies: dict[str, int]) -> Optional[dict[str, float]]:
    if type(frequencies) == dict and all(type(i) == str for i in frequencies.keys()) == True and all(
            type(c) == int for c in frequencies.values()) == True:
        word_sum = sum(list(frequencies.values()))
        words_tf = {}
        for word, frequency in frequencies.items():
            words_tf[word] = frequency/word_sum
        return words_tf

    """
    Calculates Term Frequency score for each word in a token sequence
    based on the raw frequency

    Parameters:
    frequencies (Dict): Raw number of occurrences for each of the tokens

    Returns:
    dict: A dictionary with tokens and corresponding term frequency score

    In case of corrupt input arguments, None is returned
    """


import math
def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> Optional[dict[str, float]]:
    if (type(term_freq) == dict and all(type(i) == str for i in term_freq.keys()) == True and all(
        type(i)  == float for i in term_freq.values()) == True and type(idf) == dict and all(
        type(i) == str for i in idf.keys()) == True and all(
        type(i) == float for i in idf.values()) == True):
        for word in term_freq.keys():
            if word not in idf.keys():
                idf[word] = math.log(47)
calculate_tfidf({'f':5.0, 't':6.7}, {'t':6.78})

    # """
    # Calculates TF-IDF score for each of the tokens
    # based on its TF and IDF scores
    #
    # Parameters:
    # term_freq (Dict): A dictionary with tokens and its corresponding TF values
    # idf (Dict): A dictionary with tokens and its corresponding IDF values
    #
    # Returns:
    # Dict: A dictionary with tokens and its corresponding TF-IDF values
    #
    # In case of corrupt input arguments, None is returned
    # """


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
