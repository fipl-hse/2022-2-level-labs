"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any, Type
import math


def type_of_elements(an_object: Any, elem_type: Any, key: Type = str, value: Any = int) -> bool:
    """
    Checks the type of elements in a collection
    """
    if all(isinstance(element, elem_type) for element in an_object):
        return True
    if isinstance(an_object, dict):
        if all(isinstance(element, key) for element in an_object.keys()) and all(
                isinstance(element, value) for element in an_object.values()
        ):
            return True
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
    if not isinstance(text, str):
        return None
    punctuation = r"""!()-[]{};:'"\,<>./?@#$%^&*_~"""
    res = ""
    for element in text:
        if element not in punctuation:
            res += element
    cleaned_text = res.lower().split()
    return cleaned_text


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
    if not (isinstance(tokens, list) and isinstance(stop_words, list) and
            tokens and
            type_of_elements(tokens, str) and type_of_elements(stop_words, str)
    ):
        return None
    tokens_cleaned = [key_word for key_word in tokens if key_word not in stop_words]
    return tokens_cleaned


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(tokens, list) and type_of_elements(tokens, str) and tokens):
        return None
    frequency_dict = {token: tokens.count(token) for token in tokens}
    return frequency_dict


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
    if not (isinstance(frequencies, dict)
            and isinstance(top, int) and top > 0
            and not isinstance(top, bool)
            and frequencies
            and type_of_elements(frequencies, tuple, str, Union[int, float])
    ):
        return None
    sorting = sorted(frequencies.items(), reverse=True, key=lambda item: item[1])
    final_sorting = sorting[:top]
    top_words = []
    for item in final_sorting:
        top_words.append(item[0])
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
    if not (isinstance(frequencies, dict) and type_of_elements(frequencies, tuple, str, int) and frequencies):
        return None
    term_freq_dict = {}
    length_of_the_text = sum(frequencies.values())
    for element in frequencies.items():
        term_freq = element[1] / length_of_the_text
        term_freq_dict[element[0]] = term_freq
    return term_freq_dict


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
    if not (
            isinstance(term_freq, dict)
            and isinstance(idf, dict)
            and type_of_elements(term_freq, tuple, str, float)
            and type_of_elements(idf, tuple, str, float)
            and term_freq
    ):
        return None
    max_idf = math.log(47 / 1)
    tfidf = {term: term_f * idf.get(term, max_idf) for term, term_f in term_freq.items()}
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
    if not (
            isinstance(doc_freqs, dict)
            and isinstance(corpus_freqs, dict)
            and type_of_elements(doc_freqs, tuple, str, int)
            and doc_freqs
            and type_of_elements(corpus_freqs, tuple, str, int)
    ):
        return None
    text_words_input = sum(doc_freqs.values())
    collection_words_input = sum(corpus_freqs.values())
    expected_freq_dict = {}
    for element in doc_freqs.items():
        all_words_text = text_words_input - element[1]
        elem_from_corpus = corpus_freqs.get(element[0], 0)
        all_words_collection = collection_words_input - elem_from_corpus
        expected_frequency = (
                (element[1] + elem_from_corpus)
                * (element[1] + all_words_text)
                / (element[1] + elem_from_corpus + all_words_text + all_words_collection)
        )
        expected_freq_dict[element[0]] = expected_frequency
    return expected_freq_dict


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
    if not (
            isinstance(expected, dict)
            and isinstance(observed, dict)
            and type_of_elements(expected, tuple, str, float)
            and type_of_elements(observed, tuple, str, int)
            and expected
            and observed
    ):
        return None
    hi_sqrt_dict = {
        element: math.pow(observed[element] - expected[element], 2) / expected[element]
        for element in expected.keys()
    }
    return hi_sqrt_dict


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
    if not (
            isinstance(chi_values, dict)
            and type_of_elements(chi_values, tuple, str, float)
            and chi_values
            and alpha in criterion.keys()
    ):
        return None
    chi_squar_limit = criterion[alpha]
    significant_words_dict = {}
    for word, chi_value in chi_values.items():
        if chi_value > chi_squar_limit:
            significant_words_dict[word] = chi_value
    return significant_words_dict
