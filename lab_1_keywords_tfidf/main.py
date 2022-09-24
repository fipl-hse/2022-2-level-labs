from typing import Optional, Union
import string
import math

def dictionary_check(dictionary: dict, possible_type: type, empty = False) -> bool:
    if isinstance(dictionary, dict):
        if dictionary == {} and empty is False:
            return False
        for key, value in dictionary.items():
            if not isinstance(key, str) or not isinstance(value, (int, possible_type)) or isinstance(value, bool):
                return False
        return True
    return False


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    if text and isinstance(text, str):
        text = text.lower()
        for element in string.punctuation:
            text = text.replace(element, '')
        tokens = [element for element in text.split()]
        print(tokens)
        return tokens


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    if tokens and isinstance(tokens, list):
        tokens_clean = [i for i in tokens if i not in stop_words]
        print(tokens_clean)
        return tokens_clean


def calculate_frequencies(tokens_clean: list[str]) -> Optional[dict[str, int]]:
    if tokens_clean and isinstance(tokens_clean, list):
        frequencies = {i: tokens_clean.count(i) for i in tokens_clean}
        print(frequencies)
        return frequencies


def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
    if dictionary_check(frequencies, float) and (not isinstance(top, bool) and isinstance(top, int) and not top <= 0):
        words = sorted(frequencies.keys(), key=lambda key: frequencies[key], reverse=True)
        top_five = words[:top]
        print(top_five)
        return top_five


def calculate_tf(frequencies: dict[str, int]) -> Optional[dict[str, float]]:
    if dictionary_check(frequencies, int):
        term_dict = {word: (numb / sum(frequencies.values())) for word, numb in frequencies.items()}
        print(term_dict)
        return term_dict


def calculate_tfidf(term_dict: dict[str, float], idf: dict[str, float]) -> Optional[dict[str, float]]:
    if dictionary_check(term_dict, float) and isinstance(idf, dict):
        tfidf_dict = {}
        for word in term_dict:
            if word not in idf.keys():
                idf[word] = math.log(47/1)
            tfidf_dict[word] = term_dict[word] * idf[word]
        print(tfidf_dict)
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