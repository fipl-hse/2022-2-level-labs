"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any, Type
import string
import math


def right_type(value: Any, expected_type: Type[Any]) -> bool:
    """
    Works like built-in isinstance but differentiates between int and bool
    """
    if expected_type is int:
        return isinstance(value, expected_type) and not isinstance(value, bool)
    return isinstance(value, expected_type)


def right_type_container(container: Union[list, tuple, set], container_type: Type[Any],
                         elements_type: Type[Any], allow_empty: bool = True) -> bool:
    """
    Checks datatype of a container and its elements
    """
    empty_check = allow_empty or container
    return bool((isinstance(container, container_type) and
                 all(right_type(i, elements_type) for i in container) and empty_check))


def right_dict(dictionary: dict, keys_type: Type[Any], values_type: Type[Any], allow_empty: bool = True) -> bool:
    """
    Checks datatype of keys and values of dictionary
    """
    empty_check = allow_empty or dictionary
    return bool(isinstance(dictionary, dict)
                and all(right_type(key, keys_type) for key in dictionary.keys())
                and all(right_type(value, values_type) for value in dictionary.values())
                and empty_check)


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
    without_punctuation = ''.join([i for i in text if i not in string.punctuation])
    return without_punctuation.lower().split()


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
    if not right_type_container(tokens, list, str) or not right_type_container(stop_words, list, str):
        return None
    return [token for token in tokens if token not in stop_words]


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not right_type_container(tokens, list, str, allow_empty=False):
        return None
    return {token: tokens.count(token) for token in tokens}


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
    if (not right_type(top, int)
            or top <= 0
            or not (right_dict(frequencies, keys_type=str, values_type=int, allow_empty=False)
                    or right_dict(frequencies, keys_type=str, values_type=float, allow_empty=False))):
        return None
    return sorted(frequencies, key=lambda x: frequencies.get(x, 0), reverse=True)[:top]


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
    if (not isinstance(frequencies, dict)
            or not right_dict(frequencies, keys_type=str, values_type=int, allow_empty=False)):
        return None
    total = sum(frequencies.values())
    return {term: occur / total for term, occur in frequencies.items()}


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
    if (not right_dict(term_freq, keys_type=str, values_type=float, allow_empty=False)
            or not right_dict(idf, keys_type=str, values_type=float)):
        return None
    tfidf_dict = {}
    number_of_documents = 47
    for term in term_freq:
        tfidf_dict[term] = idf.get(term, math.log(number_of_documents / 1)) * term_freq[term]
    return tfidf_dict


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
    if (not right_dict(doc_freqs, keys_type=str, values_type=int, allow_empty=False)
            or not right_dict(corpus_freqs, keys_type=str, values_type=int)):
        return None
    new_freq = {}
    for token in doc_freqs:
        given_token_freq_in_given_document = doc_freqs[token]
        given_token_freq_in_other_documents = corpus_freqs.get(token, 0)
        other_token_freq_in_given_document = sum(corpus_freqs.values()) - given_token_freq_in_other_documents
        other_token_freq_in_other_documents = sum(doc_freqs.values()) - given_token_freq_in_given_document

        res = (((given_token_freq_in_given_document + other_token_freq_in_other_documents)
                * (given_token_freq_in_given_document + given_token_freq_in_other_documents))
               / (given_token_freq_in_given_document + other_token_freq_in_other_documents
                  + other_token_freq_in_given_document + given_token_freq_in_other_documents))

        new_freq[token] = res
    return new_freq


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
    if not (right_dict(expected, keys_type=str, values_type=float, allow_empty=False)
            and right_dict(observed, keys_type=str, values_type=int, allow_empty=False)):
        return None
    new_freq = {}
    for token in expected:
        new_freq[token] = ((observed[token] - expected[token]) ** 2) / expected[token]
    return new_freq


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
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if (not right_dict(chi_values, keys_type=str, values_type=float, allow_empty=False)
            or not isinstance(alpha, float)
            or not criterion.get(alpha)):
        return None
    return {word: value for word, value in chi_values.items() if value > criterion[alpha]}
