import math

"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union


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
        text_without_punctuation = ""
        for i in text:
            if i.isalpha():
                text_without_punctuation += i
            elif i == ' ':
                text_without_punctuation += i
            elif i.isdigit():
                i = str(i)
                text_without_punctuation += i
        text_low = text_without_punctuation.lower()
        final_text = text_low.split()
        return final_text
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
    if isinstance(tokens, list) and isinstance(stop_words, list):
        if stop_words != []:
            for a in tokens:
                if isinstance(a, str):
                    for b in stop_words:
                        if isinstance(b, str):
                            final_text = []
                            for i in tokens:
                                if i not in stop_words:
                                    final_text += [i]
                            return final_text
                        else:
                            return None
                else:
                    return None
        else:
            final_text = []
            for i in tokens:
                if i not in stop_words:
                    final_text += [i]
            return final_text
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
    if isinstance(tokens, list) and tokens != []:
        for a in tokens:
            if isinstance(a, str):
                for j in tokens:
                    if type(j) == str:
                        frequency_dictionary = {}
                        for i in tokens:
                            frequency_dictionary.update({i: tokens.count(i)})
                        return frequency_dictionary
                    else:
                        return None
            else:
                return None
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
    if isinstance(frequencies, dict) and isinstance(top, int) and isinstance(top, bool) == False and top > 0:
        for a in frequencies.keys():
            if isinstance(a, str):
                for b in frequencies.values():
                    if isinstance(b, int) or isinstance(b, float):
                        sort_dictionaries_lst = sorted(frequencies.values(), reverse=True)
                        sort_dictionaries_dct = {}
                        for i in sort_dictionaries_lst:
                            for j in frequencies.keys():
                                if frequencies[j] == i:
                                    sort_dictionaries_dct[j] = i
                        top_lst = []
                        j = 0
                        for i in sort_dictionaries_dct.keys():
                            top_lst += [i]
                            j += 1
                            if j == top:
                                break
                        return top_lst
                    else:
                        return None
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
    if isinstance(frequencies, dict):
        if None in frequencies.keys():
            return None
        else:
            for k in frequencies.keys():
                if isinstance(k, str):
                    for v in frequencies.values():
                        if isinstance(v, int):
                            all_words = 0
                            for i in frequencies.values():
                                all_words += i
                            for k1, v1 in frequencies.items():
                                frequencies[k1] = v1 / all_words
                            return frequencies
                        else:
                            return None
                else:
                    return None
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
    if isinstance(term_freq, dict) and isinstance(idf, dict):
        if None in term_freq.keys():
            return None
        else:
            for k, v in term_freq.items():
                if isinstance(k, str) and isinstance(v, float):
                    if idf == {}:
                        for k1, v1 in term_freq.items():
                            term_freq[k1] = v1 * math.log(47 / 1)
                        return term_freq
                    else:
                        for k0, v0 in term_freq.items():
                            if isinstance(k0, str) and isinstance(v0, float):
                                for k2, v2 in term_freq.items():
                                    if k2 in idf.keys():
                                        term_freq[k2] = v2 * idf[k2]
                                    else:
                                        term_freq[k2] = v2 * math.log(47 / 1)
                                return term_freq
                            else:
                                return None
                else:
                    return None

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


