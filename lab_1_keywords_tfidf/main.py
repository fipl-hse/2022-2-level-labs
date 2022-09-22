"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import math


def type_of_elements(object, elem_type, key=str, value=int):
    if all(isinstance(element, elem_type) for element in object):
        return True
    elif isinstance(object, dict):
        if all(isinstance(element, key) for element in object.keys()) and \
                all(isinstance(element, value) for element in object.values()):
            return True
    else:
        return False


def open_document():
    with open("assets\Дюймовочка.txt", "r", encoding="utf-8") as file:
        return file.read()


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
    else:
        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
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
    if not isinstance(tokens, list) and not isinstance(stop_words, list) and not tokens:
        return None
    else:
        if type_of_elements(tokens, str) and type_of_elements(stop_words, str):
            tokens_cleaned = [key_word for key_word in tokens if key_word not in stop_words]
            return tokens_cleaned
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
    if not isinstance(tokens, list) and not type_of_elements(tokens, str) and not tokens:
        return None
    else:
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
    if isinstance(frequencies, dict) and type(top) is int and top > 0 and frequencies:
        if type_of_elements(frequencies, tuple, str, int | float):
            sorting = sorted(frequencies.items(), reverse=True, key=lambda item: item[1])
            final_sorting = sorting[:top]
            top_words = []
            for item in final_sorting:
                top_words.append(item[0])
            return top_words
        else:
            return None
    else:
        return None


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
    if isinstance(frequencies, dict) and type_of_elements(frequencies, tuple, str, int) and frequencies:
        words_in_text = clean_and_tokenize(open_document())
        term_freq_dict = {}
        for element in frequencies.items():
            term_freq = element[1] / len(words_in_text)
            term_freq_dict[element[0]] = term_freq
        return term_freq_dict
    else:
        return None


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
    if isinstance(term_freq, dict) and isinstance(idf, dict) and type_of_elements(term_freq, tuple, str, float)\
            and type_of_elements(idf, tuple, str, float) and term_freq:
        tfidf_dict = {}
        all_texts = 47
        docs_with_word = 0
        for element in term_freq.items():
            if element not in idf.items():
                tfidf = math.log(abs(all_texts)/(abs(docs_with_word) + 1))
                tfidf_dict[element[0]] = tfidf
            tfidf = element[1] * idf[element[0]]
            tfidf_dict[element[0]] = tfidf
        return tfidf_dict
    else:
        return None


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
    if isinstance(doc_freqs, dict) and isinstance(corpus_freqs, dict) and type_of_elements(doc_freqs, tuple, str, int) \
            and type_of_elements(corpus_freqs, tuple, str, int):
        collection_words_input = sum(doc_freqs.values())
        other_collection_words_input = sum(corpus_freqs.values())
        expected_freq_dict = {}
        for element in doc_freqs.items():
            l = collection_words_input - element[1]
            m = other_collection_words_input - corpus_freqs[element[0]]
            expected_frequency = (element[1] + corpus_freqs[element[0]]) * (element[1] + l) / (element[1] + corpus_freqs[element[0]] + l + m)
            expected_freq_dict[element[0]] = expected_frequency
        return expected_freq_dict
    else:
        return None


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
