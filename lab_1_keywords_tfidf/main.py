"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from math import log


def check_input(user_input, input_type: type, accept_float: bool) -> bool:
    """
    Checks weather the object is right type
    """
    if input_type == list:
        if isinstance(user_input, list):
            if user_input:
                for element in user_input:
                    if not isinstance(element, str):
                        return False
            return False
        return True

    if input_type == dict:
        if isinstance(user_input, dict):
            if user_input:
                for k, v in user_input.items():
                    if isinstance(k, str) and (isinstance(v, int) or isinstance(v, float)):
                        if isinstance(v, float) and accept_float is False:
                            return False
                    return False
            return False
        return True

    if input_type == int:
        if isinstance(user_input, int):
            if isinstance(user_input, bool):
                return False
            return False
        return True

    if input_type == str:
        if isinstance(user_input, str):
            if not user_input:
                return False
        return True


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    if check_input(text, str, False):
        punctuation = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
        my_text = ''
        for i in text.lower().replace('\n', ' '):
            if i not in punctuation:
                my_text += i
        my_text = my_text.split()
        return my_text
    else:
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
    if check_input(tokens, list, False) and check_input(stop_words, list, False):
        for token in tokens:
            if token not in stop_words:
                my_tokens.append(token)
        return my_tokens
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
    if check_input(tokens, list, False):
        if tokens:
            my_dict = {token: tokens.count(token) for token in tokens}
            return my_dict
    else:
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
    if check_input(frequencies, dict, True) and check_input(top, int, False):
        my_frequencies = frequencies
        my_top_list = []
        if top <= len(frequencies.keys()):
            for i in range(top):
                top_token = max(my_frequencies, key=my_frequencies.get)
                my_top_list.append(top_token)
                del my_frequencies[top_token]
            return my_top_list
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
    if check_input(frequencies, dict, False):
        tf_dict = {word: (frequency / sum(frequencies.values())) for word, frequency in frequencies.items()}
        return tf_dict
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
    if check_input(term_freq, dict, True) and check_input(idf, dict, True):
        tfidf_dict = {}
        for word, freq in term_freq.items():
            if idf.get(word) is None:
                idf_score = log(47 / (0 + 1))
            else:
                idf_score = idf.get(word)
            tfidf_dict[word] = term_freq.get(word) * idf_score
        return tfidf_dict
    else:
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
    if check_input(doc_freqs, dict, False) and check_input(corpus_freqs, dict, False):
        dict_exp_freqs = {}
        for word, freq in doc_freqs.items():
            dfw = doc_freqs.get(word)
            cfw = corpus_freqs.get(word)
            dict_exp_freqs[word] = ((dfw + cfw) * (dfw + sum(doc_freqs.values()) - dfw)) / (dfw + cfw + dfw + sum(doc_freqs.values()) - dfw + sum(corpus_freqs.values()) - cfw)
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