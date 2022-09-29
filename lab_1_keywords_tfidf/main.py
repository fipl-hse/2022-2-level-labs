"""
Lab 1
Extract keywords based on frequency related metrics
"""
import copy
from typing import Optional, Union
import string
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
    tokens_is_list = False
    tokens_are_str = True
    if isinstance(tokens, list):
        tokens_is_list = True
        for item in tokens:
            if not isinstance(item, str):
                tokens_are_str = False

    stop_words_is_list = False
    stop_words_are_str = True
    if isinstance(stop_words, list):
        stop_words_is_list = True
        for item in stop_words:
            if not isinstance(item, str):
                stop_words_are_str = False

    if tokens_is_list and stop_words_is_list and tokens_are_str and stop_words_are_str:
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
    tokens_list_is_list = False
    items_are_str = True
    if isinstance(tokens, list):
        tokens_list_is_list = True
        for item in tokens:
            if not isinstance(item, str):
                items_are_str = False

    if tokens_list_is_list and items_are_str:
        unique_tokens = set(tokens)
        unique_tokens_list = list(unique_tokens)
        counting_list = []
        counting = 0
        for item1 in unique_tokens_list:
            for item in tokens:
                if item1 == item:
                    counting += 1
            counting_list.append(counting)
            counting = 0
        tokens_dict = {unique_tokens_list[i]: counting_list[i] for i in range(len(unique_tokens_list))}
        return tokens_dict
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
    frequencies_is_dict = False
    keys_are_str = True
    values_are_int_or_float = True
    top_is_int = False
    if isinstance(top, int) and top > 0:
        top_is_int = True
        if isinstance(frequencies, dict) and len(frequencies) != 0:
            frequencies_is_dict = True
            for item in frequencies.keys():
                if not isinstance(item, str):
                    keys_are_str = False
            for item1 in frequencies.values():
                if not isinstance(item1, (int, float)):
                    values_are_int_or_float = False

    frequencies_copy = copy.deepcopy(frequencies)

    if frequencies_is_dict and keys_are_str and values_are_int_or_float and top_is_int and not isinstance(top, bool):
        key_lst = []
        sorted_values = sorted(frequencies_copy.values())
        sorted_values_list = list(sorted_values)
        sorted_values_list.reverse()
        for value in sorted_values_list:
            for key in frequencies_copy.keys():
                if frequencies_copy[key] == value:
                    key_lst.append(key)
                    frequencies_copy.pop(key)
                    break
        top_n = key_lst[:top]

        return top_n
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
    frequencies_is_dict = False
    keys_are_str = True
    values_are_int = True
    if isinstance(frequencies, dict):
        frequencies_is_dict = True
        for item in frequencies.keys():
            if not isinstance(item, str):
                keys_are_str = False
        for item1 in frequencies.values():
            if not isinstance(item1, int):
                values_are_int = False

    if frequencies_is_dict and keys_are_str and values_are_int:
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
    if isinstance(term_freq, dict) and len(term_freq) != 0:
        for item in term_freq.items():
            if not isinstance(item[0], str) or not isinstance(item[1], float):
                return None
        if isinstance(idf, dict):
            for item in idf.items():
                if not isinstance(item[0], str) or not isinstance(item[1], float):
                    return None
            tfidf_list = []
            for item_key in term_freq.keys():
                if item_key in list(idf.keys()):
                    for item1 in idf.keys():
                        if item_key == item1:
                            tfidf = term_freq[item_key] * idf[item1]
                            tfidf_list.append(tfidf)
                else:
                    tfidf = term_freq[item_key] * math.log(47 / 1)
                    tfidf_list.append(tfidf)

            keys = term_freq.keys()
            keys_list = list(keys)

            tfidf_dict = {keys_list[i]: tfidf_list[i] for i in range(len(keys_list))}
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

    if isinstance(doc_freqs, dict):
        for item in doc_freqs.items():
            if not isinstance(item[0], str) or not isinstance(item[1], int):
                return None

        if isinstance(corpus_freqs, dict):
            for item in corpus_freqs.items():
                if not isinstance(item[0], str) or not isinstance(item[1], int):
                    return None

            total_doc = 0
            for value in doc_freqs.values():
                total_doc += value

            total_corpus = 0
            for value in corpus_freqs.values():
                total_corpus += value

            expected_list = []
            for item1 in doc_freqs:
                doc_entry = doc_freqs[item1]
                corpus_entry = corpus_freqs[item1]
                other_words_doc_entry = total_doc - doc_entry
                other_words_corpus_entry = total_corpus - corpus_entry
                expected = ((doc_entry + corpus_entry) * (doc_entry + other_words_doc_entry) /
                            (doc_entry + corpus_entry + other_words_doc_entry + other_words_corpus_entry))
                expected_list.append(expected)

            keys = doc_freqs.keys()
            keys_list = list(keys)

            expected_dict = {keys_list[i]: expected_list[i] for i in range(len(keys_list))}
            return expected_dict
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
