"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import math


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
        clean_text_list = []
        for element in text:
            if element not in '.,?!;:-()—"\'%#@*&$+=[]{}<>^~`|/':
                clean_text_list.append(element)
        if clean_text_list[0] == '':
            clean_text_list.pop(0)
        if clean_text_list[-1] == '':
            clean_text_list.pop(-1)
        clean_text = ''.join(clean_text_list)
        clean_text = clean_text.replace('  ', ' ')
        clean_text = clean_text.lower()
        clean_text_list_of_words = clean_text.split()
        return clean_text_list_of_words
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
        for i in tokens:
            if not isinstance(i, str):
                return None
        for i in stop_words:
            if not isinstance(i, str):
                return None
        clean_tokens = []
        for element in tokens:
            if element not in stop_words:
                clean_tokens.append(element)
        return clean_tokens
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
    if isinstance(tokens, list):
        for i in tokens:
            if not isinstance(i, str):
                return None
        words_and_frequencies = {}
        for element in tokens:
            if element not in words_and_frequencies:
                count_element = tokens.count(element)
                words_and_frequencies[element] = count_element
        return words_and_frequencies
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
    if isinstance(frequencies, dict) and isinstance(top, int) and not isinstance(top, bool):
        for i in frequencies.keys():
            if not isinstance(i, str):
                return None
        freq_val = list(frequencies.values())
        for i in freq_val:
            if not isinstance(i, (int, float)):
                return None
        reversed_dict = dict(sorted(list(frequencies.items()), key=lambda c: c[1], reverse=True))
        top_n = []
        list_of_keys = list(reversed_dict.keys())
        if len(list(reversed_dict.keys())) < top:
            for i in enumerate(list_of_keys):
                top_n.append(list_of_keys[i[0]])
        else:
            for i in range(top):
                top_n.append(list_of_keys[i])
        if top_n:
            return top_n
    return None
get_top_n({'g':5},2)

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
    if frequencies and isinstance(frequencies, dict):
        for i in frequencies.keys():
            if not isinstance(i, str):
                return None
        for i in frequencies.values():
            if not isinstance(i, int):
                return None
        word_sum = sum(list(frequencies.values()))
        words_tf = {}
        for word, frequency in frequencies.items():
            words_tf[word] = frequency/word_sum
        return words_tf
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
    if term_freq and isinstance(term_freq, dict) and isinstance(idf, dict):
        for i in term_freq.keys():
            if not isinstance(i, str):
                return None
        for i in term_freq.values():
            if not isinstance(i, float):
                return None
        for i in idf.keys():
            if not isinstance(i, str):
                return None
        for i in idf.values():
            if not isinstance(i, float):
                return None
        for word in term_freq.keys():
            if word not in idf.keys():
                idf[word] = math.log(47)
        tfidf_dict = {}
        for word, if_value in term_freq.items():
            tfidf_dict[word] = if_value * idf[word]
        return tfidf_dict
    return None


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
    if doc_freqs and isinstance(doc_freqs, dict) and isinstance(corpus_freqs, dict):
        for i in doc_freqs.keys():
            if not isinstance(i, str):
                return None
        for i in doc_freqs.values():
            if not isinstance(i, int):
                return None
        for i in corpus_freqs.keys():
            if not isinstance(i, str):
                return None
        for i in corpus_freqs.values():
            if not isinstance(i, int):
                return None
        expected = {}
        for word, frequency in doc_freqs.items():
            if word not in corpus_freqs.keys():
                corpus_freqs[word] = 0
            word_in_doc = frequency
            word_in_corp = corpus_freqs[word]
            other_words_in_doc = sum(list(doc_freqs.values())) - word_in_doc
            other_words_in_corp = sum(list(corpus_freqs.values())) - word_in_corp
            exp_freq = ((word_in_doc+word_in_corp)*(word_in_doc+other_words_in_doc) /
                        (word_in_doc+word_in_corp+other_words_in_doc+other_words_in_corp))
            expected[word] = exp_freq
        return expected
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
    if expected and isinstance(expected, dict) and observed and isinstance(observed, dict):
        for i in expected.keys():
            if not isinstance(i, str):
                return None
        for i in expected.values():
            if not isinstance(i, float):
                return None
        for i in observed.keys():
            if not isinstance(i, str):
                return None
        for i in observed.values():
            if not isinstance(i, int):
                return None
        chi_values_dict = {}
        for word, frequency in expected.items():
            chi_square = ((observed[word]-frequency)**2)/frequency
            chi_values_dict[word] = chi_square
        return chi_values_dict
    return None


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
    if chi_values and isinstance(chi_values, dict) and isinstance(alpha, float):
        for i in chi_values.keys():
            if not isinstance(i, str):
                return None
        for i in chi_values.values():
            if not isinstance(i, float):
                return None
        criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
        if alpha in criterion.keys():
            words_dict = {}
            for word, chi in chi_values.items():
                if chi > criterion[alpha]:
                    words_dict[word] = chi
            return words_dict
    return None
