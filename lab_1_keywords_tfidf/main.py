"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import string


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    if type(text) == str:
        text = text.lower()
        for a in string.punctuation:
            if a in text:
                text = text.replace(a, '')
        text = text.strip().split()
        return text
    else:
        return None



def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    if all(isinstance(x, str) for x in tokens) and all(isinstance(x, str) for x in stop_words):
        b = 0
        while b < len(tokens):
            for c in stop_words:
                if tokens[b] == c:
                    tokens.remove(c)
                    b -= 1
            b += 1
        return tokens
    else:
        return None


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    if type(tokens) == list and len(tokens) != 0:
        d = {}
        for i in tokens:
            if type(i) == str:
                if i in d.keys():
                    d[i] = 1 + d[i]
                else:
                    d[i] = 1
            else:
                return None
        return d
    else:
        return None


def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
    if top.isnumeric():
        i = round(float(top))
        frequencies = sorted(frequencies.items(), key=lambda item: item[1])
        a = frequencies[::-1]
        d = a[:i]
        return d
    elif top.isalpha():
        return None

    #"""
    #Extracts a certain number of most frequent tokens

    #Parameters:
    #frequencies (Dict): A dictionary with tokens and
    #its corresponding frequency values
    #top (int): Number of token to extract

    #Returns:
    #List[str]: Sequence of specified length
    #consisting of tokens with the largest frequency

    #In case of corrupt input arguments, None is returned
    #"""
    #pass


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
