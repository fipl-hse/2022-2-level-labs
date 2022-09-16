"""
Lab 1
Extract keywords based on frequency related metrics
"""
import re
import string
import math
from typing import Optional, Union

#If x is None:
def for_i_type_checker(collection, type_of_instance):
    return all(map(lambda x: isinstance(x, type_of_instance), collection))

def clean_and_tokenize(text: str) -> Optional[list[str]]:
    print(isinstance(True, int))
    if not isinstance(text, str):
        return None

    for item in string.punctuation+"“”":
            text = text.replace(item, "")

    text = re.sub(r"\s{2,}", " ", text.lower())
    tokens = text.split()
    return tokens


    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    pass


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:

    condition_1 = isinstance(tokens, list) and tokens != (None and False)
    condition_2 = for_i_type_checker(tokens, str) and for_i_type_checker(stop_words, str)
    condition_3 = (stop_words == [] or isinstance(stop_words, list))

    if condition_1 and condition_2 and condition_3:
        for word in stop_words:
            while word in tokens:
                tokens.remove(word)
        return tokens
    else:
        return None

    """
    Excludes stop words from the token sequence

    Parameters:
    tokens (List[str]): Original token sequence
    stop_words (List[str]: Tokens to exclude

    Returns:
    List[str]: Token sequence that does not include stop words

    In case of corrupt input arguments, None is returned
    """
    pass


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    condit_1 = isinstance(tokens, list) and tokens != (None and False)
    condit_2 = (for_i_type_checker(tokens, str))

    if condit_1 and condit_2:
        frequencies = {i: tokens.count(i) for i in frozenset(tokens)}
        return frequencies

    else:
        return None
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    pass


def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:

    key_value = list(frequencies.items()) # Подумай как эту функцию потом укоротить и улучшить, все пустые словари -- некорректные значения?

    cond_1 = isinstance(frequencies, dict) and isinstance(top, int)
    cond_2 = for_i_type_checker(list(frequencies.keys()), str)
    cond_3 = for_i_type_checker(list(frequencies.values()), int)
    cond_4 = frequencies != (None and False)

    if cond_1 and cond_2 and cond_3 and cond_4:
        slice_index = top - 1
        #print(sorted(key_value, key=lambda point: point[1]))
        # сортируем по второму значению кортежа
        value_key = [(number, word) for word, number in key_value]
        value_key.sort(reverse=True) #косяк из поссибл

        if top <= len(value_key):
            for pair in range(top):
                value_key[pair] = value_key[pair][1]
                if pair == slice_index:
                    top_n = value_key[:slice_index:]
            return top_n
        else:
            for pair in range(top):
                value_key[pair] = value_key[pair][1]
            return top_n



    else:
        return None
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
    pass

def calculate_tf(frequencies: dict[str, int]) -> Optional[dict[str, float]]:

    keys = list(frequencies.keys())
    values = list(frequencies.values())

    c_1 = isinstance(frequencies, dict)
    c_2 = for_i_type_checker(keys, str)
    c_3 = for_i_type_checker(values, int)

    if c_1 and c_2 and c_3:

        amount_of_words = sum(i for i in values)
        tf_dict = {k:v/amount_of_words for k,v in list(frequencies.items())}
        return tf_dict

    else:
        return None

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
    c_1 = isinstance(term_freq, dict) and isinstance(idf, dict)
    c_2 = for_i_type_checker(list(term_freq.keys()), str) and for_i_type_checker(list(term_freq.values()), float)
    c_3 = for_i_type_checker(list(idf.keys()), str) and for_i_type_checker(list(idf.values()), float)

    keys_values_from_term_freq = list(term_freq.items())

    if c_1 and c_2 and c_3:
        tfidf_dict = {word: (tf*idf.get(word) if word != None else math.log(tf*47))
                      for word, tf in keys_values_from_term_freq}
        return tfidf_dict

    else:
        return None
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

    doc_freq_keys = list(doc_freqs.keys())
    corp_freq_keys = list(corpus_freqs.keys())

    c_1 = isinstance(doc_freqs, dict) and isinstance(corpus_freqs, dict)
    c_2 = for_i_type_checker(doc_freq_keys, str) and for_i_type_checker(list(doc_freqs.values()), int)
    c_3 = for_i_type_checker(corp_freq_keys, str) and for_i_type_checker(list(corpus_freqs.values()), int) # Просят флоат, а верно инт (даже без флоат)

    if c_1 and c_2 and c_3:

        j = lambda x: doc_freqs.get(x)
        k = lambda x: corpus_freqs.get(x)
        l = lambda x: ((sum(j(x) for x in doc_freq_keys)) - j(x))
        m = lambda x: ((sum(k(x) for x in corp_freq_keys)) - k(x))

        formula_expected_fr = lambda x: ((j(x) + k(x)) * (j(x) + l(x)))/(j(x) + k(x) + l(x) + m(x))

        expected = {word : formula_expected_fr(word) for word in doc_freq_keys}

        return expected

    else:
        return None


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
    list_observed_keys = list(observed.keys())

    c_1 = isinstance(expected, dict) and isinstance(observed, dict)
    c_2 = for_i_type_checker(list(expected.keys()), str) and for_i_type_checker(list(expected.values()), float)
    c_3 = for_i_type_checker(list_observed_keys, str) and for_i_type_checker(list(observed.values()), int)

    if c_1 and c_2 and c_3:

        formula_chi_2 = lambda x: (pow(((observed.get(x)) - expected.get(x)), 2))/expected.get(x)
        chi_values = {word : formula_chi_2(word) for word in list_observed_keys}
        return chi_values

    else:
        return None
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

    chi_keys = list(chi_values.keys())
    CRITERION = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    critical_point = CRITERION.get(alpha)

    c_1 = isinstance(chi_values, dict) and isinstance(alpha, float) and (alpha == 0.05 or alpha == 0.01 or alpha == 0.001)
    c_2 = for_i_type_checker(chi_keys, str) and for_i_type_checker(list(chi_values.values()), float)

    if c_1 and c_2:
        significant_words = {(word if chi_values.get(word) > critical_point else None) : chi_values.get(word)
                             for word in chi_keys}
        del significant_words[None]
        return significant_words
    else:
        return None
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
