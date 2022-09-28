"""
Lab 1
Extract keywords based on frequency related metrics
"""
import math
from typing import Optional, Union
from string import punctuation
# def dict_check():
#     if isinstance(dictionary, dict) and all(isinstance(k, str) for k in dictionary.keys() and )
#         return False
#     if not smth:
#         return False
#     for k, v in smth.items():
#         if not isinstance(k, str) and :
#             return



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
        # punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for symbol in punctuation:
            text = text.replace(symbol, '')
            stripped = text.lower().split()

        return stripped

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

    if isinstance(tokens, list) and isinstance(stop_words, list) and all(isinstance(token, str) for token in tokens):
        no_stopwords = [word for word in tokens if word not in stop_words]
        return no_stopwords

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

    if isinstance(tokens, list) and tokens and all(isinstance(el, str) for el in tokens):

        dictionary = {}
        for el in tokens:
            if el in dictionary:
                dictionary[el] += 1
            else:
                dictionary[el] = 1
        return dictionary

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
    if not(isinstance(frequencies, dict) and frequencies and isinstance(top, int)
           and not isinstance(top, bool) and top > 0):
        return None
    for key, val in frequencies.items():
        if not ((key, str) and isinstance(val, (float, int))):
            return None
        # sorted_w = [i for i ]
        dict_len = len(frequencies)
        if top <= dict_len:
            top_lst = [i for i, v in sorted(frequencies.items(), key=lambda para: para[1], reverse=True)[:top]]
        else:
            top_lst = [i for i, v in sorted(frequencies.items(), key=lambda para: para[1], reverse=True)]

        return top_lst


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

    if not(isinstance(frequencies, dict) and all(isinstance(key, str) for key in frequencies.keys())
            and all(isinstance(value, (int, float)) for value in frequencies.values())):
        return None
    count_values = sum(frequencies.values())
    tf_dict = {k: (v/count_values) for k, v in frequencies.items()}
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
    if not(isinstance(term_freq, dict) and term_freq and isinstance(idf, dict)
           and all(isinstance(k, str) for k in idf.keys()) and all(isinstance(v, float) for v in idf.values())):
        return None
    for k, v in term_freq.items():
        if not (isinstance(k, str) and isinstance(v, float)):
            return None
        tfidf_d = {}
        for word in term_freq:
            if word not in idf:
                tfidf_d[word] = math.log(47/1)
            tfidf_d[word] = term_freq[word]*idf[word]
    return tfidf_d


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