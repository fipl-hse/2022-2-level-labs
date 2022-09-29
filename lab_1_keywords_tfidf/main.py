"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import math
from string import punctuation


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
    if isinstance(tokens, list) and isinstance(stop_words, list):
        if all(isinstance(i, str) for i in tokens) and all(isinstance(i, str) for i in stop_words):
            no_stop_words = []
            for word in tokens:
                if word not in stop_words:
                    no_stop_words.append(word)
            return no_stop_words
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
    if isinstance(tokens, list) and tokens != [] and all(isinstance(i, str) for i in tokens):
        token_freq = {}
        for word in tokens:
            token_freq[word] = token_freq.get(word, 0) + 1
        return token_freq
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
    if isinstance(frequencies, dict) and isinstance(top, int) and top > 0 and not isinstance(top, bool):
        if frequencies and all(isinstance(k, str) for k in frequencies.keys()) \
                and all(isinstance(v, (int, float)) for v in frequencies.values()):
            top_list = list(keys for keys, values in sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
            return top_list[:top]
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
    if isinstance(frequencies, dict) and frequencies \
            and all(isinstance(k, str) for k in frequencies.keys()) \
            and all(isinstance(v, int) for v in frequencies.values()):
        tf_dic = {}
        n_d = sum(list(frequencies.values()))
        for key, value in frequencies.items():
            tf_dic[key] = value / n_d
        return tf_dic
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
        if term_freq and all(isinstance(k, str) for k in term_freq.keys()) \
                and all(isinstance(v, float) for v in term_freq.values()):
            if all(isinstance(k, str) for k in idf.keys()) \
                    and all(isinstance(v, float) for v in idf.values()):
                tfidf_dic = {}
                for key in term_freq.keys():
                    if key not in idf.keys():
                        idf[key] = math.log(47 / 1)
                    tfidf_dic[key] = term_freq[key] * idf[key]
                # else:
                #     tfidf_dic[key] = term_freq[key] * idf[key]
                return tfidf_dic
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
    if isinstance(doc_freqs, dict) and isinstance(corpus_freqs, dict) \
            and doc_freqs and all(isinstance(k, str) for k in doc_freqs) \
            and all(isinstance(v, int) for v in doc_freqs.values()):
        if all(isinstance(k, str) for k in corpus_freqs.keys()) \
                and all(isinstance(v, int) for v in corpus_freqs.values()):
            expected_dic = {}
            for keys in doc_freqs.keys():
                if keys not in corpus_freqs:
                    corpus_freqs[keys] = 0
                value_doc = doc_freqs[keys]
                value_corpus = corpus_freqs[keys]
                except_word_doc = sum(list(doc_freqs.values())) - value_doc
                except_word_corpus = sum(list(corpus_freqs.values())) - value_corpus
                formula = ((value_doc + value_corpus) * (value_doc + except_word_doc)) / \
                          (value_doc + value_corpus + except_word_doc + except_word_corpus)
                expected_dic[keys] = formula
            return expected_dic
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
    if isinstance(expected, dict) and isinstance(observed, dict) \
            and expected and all(isinstance(k, str) for k in expected.keys()):
        if all(isinstance(v, float) for v in expected.values()) \
                and observed and all(isinstance(k, str) for k in observed.keys()) \
                and all(isinstance(v, int) for v in observed.values()):
            chi_dic = {}
            for key in observed.keys():
                expected_value = expected[key]
                observed_value = observed[key]
                chi_value = ((observed_value - expected_value) ** 2) / expected_value
                chi_dic[key] = chi_value
            return chi_dic
    return None


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
    if isinstance(chi_values, dict) and isinstance(alpha, float) \
            and chi_values and all(isinstance(k, str) for k in chi_values.keys()) \
            and all(isinstance(v, float) for v in chi_values.values()):
        if alpha in (0.05, 0.01, 0.001):
            significant_dic = {}
            criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
            alpha = criterion.get(alpha)
            for key, value in chi_values.items():
                if value >= alpha:
                    significant_dic[key] = value
            return significant_dic
    return None
