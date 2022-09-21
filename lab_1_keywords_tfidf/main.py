"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from math import log


def check_list(user_input, elements_type: type, can_be_empty: bool) -> bool:
    if isinstance(user_input, list):
        if user_input == [] and can_be_empty is False:
            return False
        for element in user_input:
            if not isinstance(element, elements_type):
                return False
        return True
    return False


def check_dict(user_input, key_type: type, value_type: type, can_be_empty: bool):
    if isinstance(user_input, dict):
        if user_input == {} and can_be_empty is False:
            return False
        for key, value in user_input.items():
            if not (isinstance(key, key_type) and isinstance(value, value_type)):
                return False
        return True
    return False


def check_positive_int(user_input):
    if isinstance(user_input, int):
        if isinstance(user_input, bool):
            return False
        if user_input <= 0:
            return False
        return True
    return False


def check_float(user_input):
    if isinstance(user_input, float):
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
    if isinstance(text, str):
        punctuation = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'''
        my_text = ''
        for i in text.lower().replace('\n', ' '):
            if i not in punctuation:
                my_text += i
        my_text = my_text.split()
        return my_text
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
    my_tokens = []
    if check_list(tokens, str, False) and check_list(stop_words, str, True):
        for token in tokens:
            if token not in stop_words:
                my_tokens.append(token)
        return my_tokens
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
    if check_list(tokens, str, False):
        if tokens:
            my_dict = {token: tokens.count(token) for token in tokens}
            return my_dict
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
    if (check_dict(frequencies, str, int, False) or check_dict(frequencies, str, float, False)) and \
            check_positive_int(top):
        my_frequencies = frequencies
        my_top_list = []
        if top >= len(frequencies):
            top = len(frequencies)
        for i in range(top):
            top_token = max(my_frequencies, key=my_frequencies.get)
            my_top_list.append(top_token)
            del my_frequencies[top_token]
        return my_top_list
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
    if check_dict(frequencies, str, int, False):
        tf_dict = {word: (frequency / sum(frequencies.values())) for word, frequency in frequencies.items()}
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
    if check_dict(term_freq, str, float, False) and check_dict(idf, str, float, True):
        tfidf_dict = {}
        for word in term_freq.keys():
            if idf.get(word) is None:
                idf_score = log(47 / (0 + 1))
            else:
                idf_score = idf.get(word)
            tfidf_dict[word] = term_freq.get(word) * idf_score
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
    if check_dict(doc_freqs, str, int, False) and check_dict(corpus_freqs, str, int, True):
        dict_exp_freqs = {}
        for word, freq in doc_freqs.items():
            except_word_doc_freq = sum(doc_freqs.values()) - freq
            corpus_freq = corpus_freqs.get(word, 0)
            except_word_corpus_freq = sum(corpus_freqs.values()) - corpus_freq
            dict_exp_freqs[word] = ((freq + corpus_freq) * (freq + except_word_doc_freq)) /\
                                   (freq + corpus_freq + except_word_doc_freq + except_word_corpus_freq)
        return dict_exp_freqs
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
    if check_dict(expected, str, float, False) and check_dict(observed, str, int, False):
        chi_dict = {}
        for word, freq in expected.items():
            chi_dict[word] = ((observed.get(word) - freq) ** 2) / freq
        return chi_dict
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
    if check_dict(chi_values, str, float, True) and check_float(alpha):
        significant_words_dict = {}
        for word, chi_value in chi_values.items():
            if chi_value >= alpha:
                significant_words_dict[word] = chi_value
        return significant_words_dict
    return None
