"""
Lab 1
Extract keywords based on frequency related metrics
"""
from re import sub
from string import punctuation
from math import log
from typing import Optional, Union, Type, Any


def for_i_empty_checker(collection: Union[set, dict, list, tuple]) -> bool:
    """
    Check if collection's items are False

    Parameters:
    collection: Union[set, dict, list, tuple]: collection, which includes some items

    Returns:
    bool: True, if items are not False (not empty, or not 0, if numbers) and False otherwise
    """
    return bool(collection and all(bool(i) for i in collection))


def my_isinstance(instance: Any, type_of_instance: Type[Any]) -> bool:
    """
    Distincts int and bool compared to built-in isinstance() function.

    Parameters:
    instance: Any
    type_of_instance: Any

    Returns:
    bool: True if instance's type is expected type, False, if instance's type is bool and expected type
    """

    if type_of_instance is not int:
        return isinstance(instance, type_of_instance)
    return bool(not isinstance(instance, bool) and isinstance(instance, int))


def for_i_type_checker(collection: Union[set, list, tuple],
                       type_of_collection: Type[Any],
                       type_of_instance: Type[Any]) -> bool:
    """
    Acts like my_isinstance for every collection's item

    Parameters:
    collection: Union[set, dict, list, tuple]
    type_of_collection: Union[set, dict, list, tuple]
    type_of_instance: Any

    Returns:
    bool: True if instance's type is expected type, False, if instance's
    type is bool and expected type is int or if expected collection's type
    doesn't equal collection's type
    """

    return (my_isinstance(collection, type_of_collection)
            and all(map(lambda x: my_isinstance(x, type_of_instance), collection)))


def is_dic_correct(dic: dict,
                   allow_false_items: bool,
                   key_type: Type[Any],
                   value_type: Type[Any]) -> bool:
    """
    Checks dictionary on being empty, having False items in keys and values,
    correspondence of keys and values to the types we expect to observe

    Parameters:
    dic: dict
    allow_false_items: bool
    key_type: Union[int, float, str, tuple]
    value_type: Any

    Returns:
    bool: True if dict is not empty,
    it's keys and values correspond to expected type
    and deprived of False items, else: False
    """

    if not my_isinstance(dic, dict):
        return False

    keys = list(dic.keys())
    values = list(dic.values())
    is_empty = bool(allow_false_items or dic)
    if is_empty:
        return (for_i_type_checker(keys, list, key_type)
                and for_i_type_checker(values, list, value_type))

    return (for_i_type_checker(keys, list, key_type) and for_i_type_checker(values, list, value_type)
            and for_i_empty_checker(keys) and for_i_empty_checker(values) and is_empty)


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """

    if not (my_isinstance(text, str) and text):
        return None

    for item in punctuation+'“”':
        text = text.replace(item, '')
    tokens = (sub(r"\s{2,}", ' ', text.lower())).split()
    return tokens


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

    if not (my_isinstance(tokens, list) and for_i_empty_checker(tokens)
            and for_i_type_checker(tokens, list, str)
            and for_i_type_checker(stop_words, list, str)
            and my_isinstance(stop_words, list)):
        return None

    for word in stop_words:
        while word in tokens:
            tokens.remove(word)
    return tokens


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """

    if not (tokens and my_isinstance(tokens, list)
            and for_i_empty_checker(tokens)
            and for_i_type_checker(tokens, list, str)):
        return None

    frequencies = {i: tokens.count(i) for i in frozenset(tokens)}
    return frequencies


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

    if not (is_dic_correct(frequencies, False, str, (int, float))
            and my_isinstance(top, int) and top > 0):
        return None

    return sorted(frequencies, key=lambda word: frequencies[word], reverse=True)[:top]


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

    if not is_dic_correct(frequencies, False, str, int):
        return None

    values = list(frequencies.values())
    amount_of_words = sum(values)
    tf_dict = {k: v / amount_of_words for k, v in list(frequencies.items())}
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

    if not (is_dic_correct(term_freq, False, str, float)
            and is_dic_correct(idf, True, str, float)):
        return None

    keys_term_freq = list(term_freq.keys())
    tfidf_dict = {word: (term_freq[word] * idf.get(word, log(47/1)))
                  for word in keys_term_freq}
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

    if not (is_dic_correct(doc_freqs, False, str, int)
            and is_dic_correct(corpus_freqs, True, str, int)):
        return None

    doc_freq_keys = list(doc_freqs.keys())
    sum_of_occ_doc = (sum(doc_freqs.values()))
    sum_of_occ_corp = (sum(corpus_freqs.values()))

    expected = {}

    for word in doc_freq_keys:
        j_occur_of_word_in_doc = doc_freqs[word]
        l_occur_doc_except = (sum_of_occ_doc - j_occur_of_word_in_doc)

        k_occur_of_word_in_corp = corpus_freqs.get(word, 0)
        m_occur_corp_except = (sum_of_occ_corp - k_occur_of_word_in_corp)

        expected_freq = (((j_occur_of_word_in_doc + k_occur_of_word_in_corp)
                          * (j_occur_of_word_in_doc + l_occur_doc_except))
                         / (j_occur_of_word_in_doc + k_occur_of_word_in_corp
                            + l_occur_doc_except + m_occur_corp_except))

        expected[word] = expected_freq
    return expected


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

    if not (is_dic_correct(expected, False, str, float) and is_dic_correct(observed, False, str, int)):
        return None

    formula_chi_2 = (lambda word: (pow((observed[word] - expected.get(word, .0)), 2)) / expected[word])
    chi_values = {word: formula_chi_2(word) for word in list(observed.keys())}
    return chi_values


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

    if not (is_dic_correct(chi_values, False, str, float) and my_isinstance(alpha, float)):
        return None
    criterion_dict = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if not criterion_dict.get(alpha, False):
        return None

    chi_keys = list(chi_values.keys())
    significant_words = {word: chi_values.get(word, .0) for word in chi_keys
                         if chi_values.get(word, .0) > criterion_dict[alpha]}
    return significant_words
