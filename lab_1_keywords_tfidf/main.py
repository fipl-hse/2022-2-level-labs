"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from collections import Counter
import string

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
        text1 = text.lower()
        text2 = text1.translate(str.maketrans('', '', string.punctuation))
        lst = text2.split()
    else:
        lst = None
    return lst



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
        lst = list()
        for item in tokens:
            if not item in stop_words:
                lst.append(item)
    else:
        lst = None
    return lst



def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    tokens_dict = Counter(tokens)
    return tokens_dict


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
    sorted_frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1]))
    keys = sorted_frequencies.keys()
    keys_lst = list(keys)
    keys_lst.reverse()
    top_n = keys_lst[:top]
    return top_n


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
    #это отношение числа вхождений некоторого слова к общему числу слов документа
    total_value = 0
    for value in frequencies.values():
        total_value += value

    term_frequencies_list = []
    for value in frequencies.values():
        term_frequency = value / total_value
        term_frequencies_list.append(term_frequency)

    keys = frequencies.keys()
    keys_list = list(keys)

    tf_dict = {keys_list[i]: term_frequencies_list[i] for i in range(len(keys_list))}
    return tf_dict


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
    tfidf_list =[]
    for item in term_freq.keys():
        for item1 in idf.keys():
            if item == item1:
                tfidf = term_freq[item] * idf[item1]
                tfidf_list.append(tfidf)

    keys = term_freq.keys()
    keys_list = list(keys)

    tfidf_dict = {keys_list[i]: tfidf_list[i] for i in range(len(keys_list))}
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
    total_doc = 0
    for value in doc_freqs.values():
        total_doc += value

    total_corpus = 0
    for value in corpus_freqs.values():
        total_corpus += value

    expected_list = []
    for item in doc_freqs:
        j = doc_freqs[item]
        k = corpus_freqs[item]
        l = total_doc - j
        m = total_corpus - k
        expected = (j + k) * (j + l) / (j + k + l + m)
        expected_list.append(expected)

    keys = doc_freqs.keys()
    keys_list = list(keys)

    expected_dict = {keys_list[i]: expected_list[i] for i in range(len(keys_list))}
    return expected_dict


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
