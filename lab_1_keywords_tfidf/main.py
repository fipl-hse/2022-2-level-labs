"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from string import punctuation
from operator import itemgetter

with open("Дюймовочка.txt", 'r', encoding="utf-8") as f:
    text = f.read()
with open("stop_words.txt", 'r', encoding="utf-8") as f:
    stop_words = f.read()


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    if not isinstance(text, str):
        return None
    else:
        for i in punctuation:
            text = text.replace(i, "")
            text = text.lower().strip()
        spisok = text.split()
        print(spisok)
        return spisok


sp = clean_and_tokenize(text)


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    tokens = stop_words.split()
    clean_spisok = []
    for i in sp:
        if i not in tokens:
            clean_spisok.append(i)
    print(clean_spisok)
    return clean_spisok


cs = remove_stop_words(sp, stop_words)


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    dictionary = {}
    for i in cs:
        if i in dictionary.keys():
            dictionary[i] += 1
        else:
            dictionary[i] = 1
    print(dictionary)
    return dictionary


slovar = calculate_frequencies(cs)


def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
    freq_dict = {}
    for word in slovar:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
    sorted_list = sorted(freq_dict.items(), key=itemgetter(1), reverse=True)
    sorted_list.most_common(5)
    # number = sorted_list[:5]
    return sorted_list


get_top_n(slovar, 5)


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
