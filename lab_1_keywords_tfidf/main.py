""""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, Any
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
    punctuation = '''!@#$%^&*()_-+=[]{};:'\\|,<.>/?"'''
    the_text = ''
    for symbol in text.lower().replace('\n', ' '):
        if symbol not in punctuation:
            the_text += symbol
    tokens = the_text.split()
    return tokens


def check_list(text: Any, elements_type: type, can_be_empty: bool) -> bool:
    if not isinstance(text, list):
        return False
    if not text and can_be_empty is False:
        return False
    for element in text:
        if not isinstance(element, elements_type):
            return False
    return True

def check_dict(text: dict, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    if not isinstance(text, dict):
        return False
    if not text and can_be_empty is False:
        return False
    for key, value in text.items():
        if not (isinstance(key, key_type) and isinstance(value, value_type)):
            return False
    return True

def check_int(text: Any) -> bool:
    if not isinstance(text, int):
        return False
    if isinstance(text, bool):
        return False
    if text <= 0:
        return False
    return True

def check_float(text: Any) -> bool:
    if not isinstance(text, float):
        return False
    else:
        return True


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
    the_tokens = []
    if not (check_list(tokens, str, False) and check_list(stop_words, str, True)):
        return None
    for token in tokens:
        if token not in stop_words:
            the_tokens.append(token)
    return the_tokens


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence
    Parameters:
    tokens (List[str]): Token sequence to count frequencies for
    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary
    In case of corrupt input arguments, None is returned
    """
    if not check_list(tokens, str, False):
        return None
    if tokens:
        my_dict = {token: tokens.count(token) for token in tokens}
    return my_dict


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
    checking_dict = check_dict(frequencies, str, int, False) or check_dict(frequencies, str, float, False)
    if not (checking_dict and check_int(top)):
        return None
    return sorted(frequencies.keys(), key=lambda key: frequencies[key], reverse=True)[:top]



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
    if not check_dict(frequencies, str, int, False):
        return None
    sum_freq = sum(frequencies.values())
    tf_dict = {word: (frequency / sum_freq) for word, frequency in frequencies.items()}
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
    if not (check_dict(term_freq, str, float, False) and check_dict(idf, str, float, True)):
        return None
    tfidf_dict = {}
    for word in term_freq.keys():
        tfidf_dict[word] = term_freq[word] * idf.get(word, log(47))
    return tfidf_dict


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
    if not (check_dict(doc_freqs, str, int, False) and check_dict(corpus_freqs, str, int, True)):
        return None
    dict_exp_freqs = {}
    for word, freq in doc_freqs.items():
        except_word_doc_freq = sum(doc_freqs.values()) - freq
        corpus_freq = corpus_freqs.get(word, 0)
        except_word_corpus_freq = sum(corpus_freqs.values()) - corpus_freq
        dict_exp_freqs[word] = ((freq + corpus_freq) * (freq + except_word_doc_freq)) /\
                                (freq + corpus_freq + except_word_doc_freq + except_word_corpus_freq)
    return dict_exp_freqs
