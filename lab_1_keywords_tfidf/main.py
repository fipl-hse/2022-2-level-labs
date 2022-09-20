"""
Lab 1
Extract keywords based on frequency related metrics
"""
import re
import string
import math
from typing import Optional, Union, Type, Any


def my_isinstance(instance, type_of_instance):
    """
    Distincts int and bool compared to built-in isinstance() function.

    Solves case: isinstance(False/True, bool) -> True
    """

    if type_of_instance is int:
        return not isinstance(instance, bool) and isinstance(instance, int)
    return isinstance(instance, type_of_instance)


def for_i_type_checker(collection, type_of_instance):

    """
    Works like my_isinstance but for every collection's item
    """

    return all(map(lambda x: my_isinstance(x, type_of_instance), collection))


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """

    if isinstance(text, str) and bool(text):
        for item in string.punctuation+'“”':
            text = text.replace(item, '')

        tokens = (re.sub(r"\s{2,}", ' ', text.lower())).split()
        return tokens
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

    condition_1 = (my_isinstance(tokens, list)
                   and
                   bool(tokens))

    condition_2 = (for_i_type_checker(tokens, str) and
                   for_i_type_checker(stop_words, str))

    condition_3 = (my_isinstance(stop_words, list))

    if condition_1 and condition_2 and condition_3:
        for word in stop_words:
            while word in tokens:
                tokens.remove(word)
        return tokens
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

    condit_1 = (my_isinstance(tokens, list)
                and
                bool(tokens))

    condit_2 = (for_i_type_checker(tokens, str))

    if condit_1 and condit_2:
        frequencies = {i: tokens.count(i) for i in frozenset(tokens)}
        return frequencies
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

    freq_items = list(frequencies.items())

    cond_1 = (my_isinstance(frequencies, dict)
              and
              my_isinstance(top, int))

    cond_2 = for_i_type_checker(list(frequencies.keys()), str)

    cond_3 = for_i_type_checker(list(frequencies.values()), int | float)

    cond_4 = bool(frequencies) and bool(top)

    if cond_1 and cond_2 and cond_3 and cond_4:
        top_n = []
        freq_items.sort(reverse=True, key=lambda pair: pair[1])
        if top > len(freq_items):
            top = len(freq_items)
        slice_index = top - 1
        for key_value in freq_items:
            top_n.append(key_value[0])
            if top_n.index(key_value[0]) == slice_index:
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

    values = list(frequencies.values())

    c_1 = (my_isinstance(frequencies, dict)
           and
           bool(frequencies))

    c_2 = for_i_type_checker(list(frequencies.keys()), str)

    c_3 = for_i_type_checker(values, int)

    if c_1 and c_2 and c_3:

        amount_of_words = sum(i for i in values)
        tf_dict = {k: v/amount_of_words for k, v in list(frequencies.items())}
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
    c_1 = (my_isinstance(term_freq, dict) and bool(term_freq)
           and
           my_isinstance(idf, dict)) and bool(idf)

    c_2 = (for_i_type_checker(list(term_freq.keys()), str)
           and
           for_i_type_checker(list(term_freq.values()), float))

    c_3 = (for_i_type_checker(list(idf.keys()), str)
           and
           for_i_type_checker(list(idf.values()), float))

    keys_values_from_term_freq = list(term_freq.items())

    if c_1 and c_2 and c_3:
        tfidf_dict = {word: (tf*idf.get(word) if word is not None else math.log(tf*47))
                      for word, tf in keys_values_from_term_freq}
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

    doc_freq_keys = list(doc_freqs.keys())
    corp_freq_keys = list(corpus_freqs.keys())

    c_1 = (my_isinstance(doc_freqs, dict) and bool(doc_freqs)
           and
           my_isinstance(corpus_freqs, dict)) and bool(corpus_freqs)
    c_2 = (for_i_type_checker(doc_freq_keys, str)
           and
           for_i_type_checker(list(doc_freqs.values()), int))
    c_3 = (for_i_type_checker(corp_freq_keys, str)
           and
           for_i_type_checker(list(corpus_freqs.values()), int))

    if c_1 and c_2 and c_3:

        j_occur_of_word_in_doc = lambda word: doc_freqs.get(word)
        k_occur_of_word_in_corp = lambda word: corpus_freqs.get(word)

        sum_of_occ_doc = (sum(j_occur_of_word_in_doc(word) for word in doc_freq_keys))
        sum_of_occ_corp = (sum(k_occur_of_word_in_corp(word) for word in corp_freq_keys))

        l_occur_doc_except = lambda word: (sum_of_occ_doc - j_occur_of_word_in_doc(word))
        m_occur_corp_except = lambda word: (sum_of_occ_corp - k_occur_of_word_in_corp(word))

        formula_expected_freq = (lambda word: ((j_occur_of_word_in_doc(word) + k_occur_of_word_in_corp(word)) *
                                (j_occur_of_word_in_doc(word) + l_occur_doc_except(word))) /
                                (j_occur_of_word_in_doc(word) + k_occur_of_word_in_corp(word) +
                                    l_occur_doc_except(word) + m_occur_corp_except(word)))

        expected = {word: formula_expected_freq(word) for word in doc_freq_keys}

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

    list_observed_keys = list(observed.keys())

    c_1 = (my_isinstance(expected, dict) and bool(expected)
           and
           my_isinstance(observed, dict)) and bool(observed)

    c_2 = (for_i_type_checker(list(expected.keys()), str)
           and
           for_i_type_checker(list(expected.values()), float))

    c_3 = (for_i_type_checker(list_observed_keys, str)
           and
           for_i_type_checker(list(observed.values()), int))

    if c_1 and c_2 and c_3:

        formula_chi_2 = lambda x: (pow(((observed.get(x)) - expected.get(x)), 2))/expected.get(x)

        chi_values = {word: formula_chi_2(word) for word in list_observed_keys}
        return chi_values
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

    chi_keys = list(chi_values.keys())
    criterion_dict = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    critical_point = criterion_dict.get(alpha)

    c_1 = (my_isinstance(chi_values, dict) and bool(chi_values)
           and
           my_isinstance(alpha, float)
           and
           critical_point)

    c_2 = (for_i_type_checker(chi_keys, str)
           and
           for_i_type_checker(list(chi_values.values()), float))

    if c_1 and c_2:
        significant_words = {word: chi_values.get(word) for word in chi_keys
                             if chi_values.get(word) > critical_point}
        return significant_words
    return None
