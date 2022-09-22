"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
from math import log


def clean_and_tokenize(text: str) -> Optional[list[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str):
        return None
    else:
        punctuation = '''!,()[]-{};:“'"”\<>./?@#$%^&*~_'''
        for i in punctuation:
            text = text.replace(i, '')
        text = text.lower().split()
        return text


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
    if not isinstance(tokens, list) and not isinstance(stop_words, list) \
            and not tokens and not all(isinstance(i, str) for i in tokens) \
            and not stop_words and not all(isinstance(i, str) for i in stop_words):
        return None
    else:
        no_stop_words = []
        for word in tokens:
            if word not in stop_words:
                no_stop_words.append(word)
        return no_stop_words


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) and not tokens and not all(isinstance(i, str) for i in tokens):
        return None
    else:
        token_freq = {}
        for word in tokens:
            token_freq[word] = token_freq.get(word, 0) + 1
        return token_freq


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
    if not isinstance(frequencies, dict) and not isinstance(top, int) and top <= 0 \
            and not frequencies and not all(isinstance(k, str) for k in frequencies.keys()) \
            and not all(isinstance(v, (int, float)) for v in frequencies.values()):
        return None
    else:
        top_dic = {}
        top_values = sorted(frequencies.values(), reverse=True)
        top_n_values = top_values[:top]
        for i in top_n_values:
            for k in frequencies.keys():
                if frequencies[k] == i:
                    top_dic[k] = frequencies[k]
        return list(top_dic.keys())


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
    if not isinstance(frequencies, dict) and not frequencies \
            and not all(isinstance(k, str) for k in frequencies.keys()) \
            and not all(isinstance(v, int) for v in frequencies.values()):
        return None
    else:
        tf_dic = {}
        n_d = sum(list(frequencies.values()))
        for k, v in frequencies.items():
            tf_dic[k] = v / n_d
        return tf_dic


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
    if not isinstance(term_freq, dict) and not isinstance(idf, dict) \
            and not term_freq and not all(isinstance(k, str) for k in term_freq.keys()) \
            and not all(isinstance(v, float) for v in term_freq.values()) \
            and not idf and not all(isinstance(k, str) for k in idf.keys()) \
            and not all(isinstance(v, float) for v in idf.values()):
        return None
    else:
        tfidf_dic = {}
        for key in term_freq.keys():
            if not idf[key] == 0:
                tfidf_dic[key] = term_freq[key] * idf[key]
            else:
                tfidf_dic[key] = term_freq[key] * log(47)
        return tfidf_dic


def calculate_expected_frequency(
        doc_freqs: dict[str, int], corpus_freqs: dict[str, float]
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
    if not isinstance(doc_freqs, dict) and not isinstance(corpus_freqs, dict) \
            and not doc_freqs and not all(isinstance(k, str) for k in doc_freqs) \
            and not all(isinstance(v, float) for v in doc_freqs.values()) \
            and not corpus_freqs and not all(isinstance(k, str) for k in corpus_freqs.keys()) \
            and not all(isinstance(v, float) for v in corpus_freqs.values()):
        return None
    else:
        expected_dic = {}
        for keys, values in doc_freqs.items():
            if keys not in corpus_freqs:
                corpus_freqs[keys] = 0
            j = doc_freqs[keys]
            k = corpus_freqs[keys]
            l = sum(list(doc_freqs.values())) - j
            m = sum(list(corpus_freqs.values())) - k
            formula = ((j + k) * (j + l)) / (j + k + l + m)
            expected_dic[keys] = formula
        return expected_dic


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
    if not isinstance(expected, dict) and not isinstance(observed, dict) \
            and not expected and not all(isinstance(k, str) for k in expected.keys()) \
            and not all(isinstance(v, float) for v in expected.values()) \
            and not observed and not all(isinstance(k, str) for k in observed.keys()) \
            and not all(isinstance(v, int) for v in observed.values()):
        return None
    else:
        chi_dic = {}
        for key in observed.keys():
            a = expected[key]
            b = observed[key]
            chi_value = ((b - a) ** 2) / a
            chi_dic[key] = chi_value
        return chi_dic


def extract_significant_words(chi_values: dict[str, float], alpha: float) -> Optional[dict[str, float]]:
    """
    Select those tokens from the token sequence that
    have a chi-squared value smaller than the criterion

    Parameters:
    chi_values (Dict): A dictionary with tokens and
    its corresponding chi-squared value
    alpha (float): Level of significance that controls critical value of chi-squared metric

    Returns:
    Dict: A dictionary with significant tokens
    and its corresponding chi-squared value

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(chi_values, dict) and not isinstance(alpha, float) \
            and not chi_values and not all(isinstance(k, str) for k in chi_values.keys()) \
            and not all(isinstance(v, float) for v in chi_values.values()) \
            and alpha not in (0.05, 0.01, 0.001):
        return None
    else:
        significant_dic = {}
        criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
        alpha = criterion.get(alpha)
        for key, value in chi_values.items():
            if value >= alpha:
                significant_dic[key] = value
        return significant_dic
