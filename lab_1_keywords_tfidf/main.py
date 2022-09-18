"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union


def type_of_elements(object, elem_type, key=str, value=int):
    if isinstance(object, dict):
        if all(isinstance(element, key) for element in object.keys()) and \
                all(isinstance(element, value) for element in object.values()):
            return True
    elif all(isinstance(element, elem_type) for element in object):
        return True
    else:
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
        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        res = " "
        for element in text:
            if element not in punctuation:
                res += element
        cleaned_text = res.lower().split()
        return cleaned_text
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
        if type_of_elements(tokens, str) and type_of_elements(stop_words, str):
            tokens_cleaned = [i for i in tokens if i not in stop_words]
            return tokens_cleaned
        else:
            return None
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
    if isinstance(tokens, list) and type_of_elements(tokens, str) and len(tokens) != 0:
        frequency_dict = {i: tokens.count(i) for i in tokens}
        return frequency_dict
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
    if isinstance(frequencies, dict) and type(top) is int and len(frequencies) > 0 and top > 0:
        if type_of_elements(frequencies, tuple, str, int | float):
            sorting = sorted(frequencies.items(), reverse=True, key=lambda item: item[1])
            final_sorting = sorting[:top]
            top_words = []
            for item in final_sorting:
                top_words.append(item[0])
            return top_words
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
    if isinstance(frequencies, dict) and type_of_elements(frequencies, tuple, str, int):
        term_freq_dict = {}
        for element in frequencies.items():
            term_freq = element[1] / len(frequencies)
            temporary_dict = {element[0]: term_freq}
            term_freq_dict.update(temporary_dict)
        return term_freq_dict
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
    if isinstance(term_freq, dict) and isinstance(idf, dict) and type_of_elements(term_freq, tuple, str, float) and type_of_elements(idf, tuple, str, float):
        tfidf_dict = {}
        for element in term_freq.items():
            tfidf = element[1] * idf[element[0]]
            new_dict = {element[0]: tfidf}
            tfidf_dict.update(new_dict)
        return tfidf_dict
    else:
        return None



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
    # collection_words_input = 0
    # for word in doc_freqs.values():
    #     collection_words_input += word
    # other_collection_words_input = 0
    # for other_word in corpus_freqs.values():
    #     other_collection_words_input += other_word
    # for element in doc_freqs:
    #     l = collection_words_input - element[1]
    #     m = other_collection_words_input
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
