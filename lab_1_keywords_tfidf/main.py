"""
Lab 1
Extract keywords based on frequency related metrics
"""
import math
from typing import Optional, Union
import string


def correct_list(list1: list, type1: type, empty=False) -> bool:
    if isinstance(list1, list):
        if list1 == [] and empty is False:
            return False
        for index in list1:
            if isinstance(index, type1) is False:
                return False
        return True
    return False


def correct_dict(dictionary: dict, type1: type, type2: type, empty=False) -> bool:
    if isinstance(dictionary, dict):
        if dictionary == {} and empty is False:
            return False
        for key, value in dictionary.items():
            if not isinstance(key, type1) or not isinstance(value, (int, type2)) or isinstance(value, bool):
                return False
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
    if not isinstance(text, str):
        return None
    lowered_text = text.lower()
    for punctuation_mark in string.punctuation:
        if punctuation_mark in lowered_text:
            lowered_text = lowered_text.replace(punctuation_mark, '')
    cleaned_text = lowered_text.strip().split()
    return cleaned_text


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
    if correct_list(tokens, str) and correct_list(stop_words, str, True):
        index = 0
        while index < len(tokens):
            for stop_word in stop_words:
                if tokens[index] is stop_word:
                    tokens.remove(stop_word)
                    index -= 1
                    break
            index += 1
        return tokens
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
    if correct_list(tokens, str):
        frequency_dict = {}
        for token in tokens:
            if not isinstance(token, str):
                return None
            if token in frequency_dict.keys():
                frequency_dict[token] = 1 + frequency_dict[token]
            else:
                frequency_dict[token] = 1
        return frequency_dict
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
    if correct_dict(frequencies, str, float) and (not isinstance(top, bool) and isinstance(top, int) and not top <= 0):
        frequencies = sorted(frequencies.items(), reverse=True, key=lambda item: item[1])
        top_list = []
        for word in frequencies:
            top_list.append(word[0])
        if len(frequencies) >= top:
            top_list = top_list[:top]
        return top_list
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
    all_words = 0
    new_dict = {}
    if correct_dict(frequencies, str, int):
        for value in frequencies.values():
            all_words += value
        for key in frequencies.keys():
            new_dict[key] = frequencies[key] / all_words
        return new_dict
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
    if correct_dict(term_freq, str, float) and correct_dict(idf, str, float, True):
        final_dict = {}
        for key_freq, value_freq in term_freq.items():
            if key_freq in idf:
                final_dict[key_freq] = value_freq * idf[key_freq]
            else:
                final_dict[key_freq] = value_freq * math.log(47)
        return final_dict
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

    if correct_dict(doc_freqs, str, int) and correct_dict(corpus_freqs, str, int, True):
        expected_freq = {}
        for key, value in doc_freqs.items():
            words_in_doc = sum(doc_freqs.values()) - value
            word_in_corpus = 0
            if key in corpus_freqs.keys():
                word_in_corpus = corpus_freqs.get(key)
            words_in_corpus = sum(corpus_freqs.values()) - word_in_corpus
            expected_freq[key] = ((value + word_in_corpus) * (value + words_in_doc)) / (
                    value + word_in_corpus + words_in_doc + words_in_corpus)
        return expected_freq
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

    if correct_dict(expected, str, float) and correct_dict(observed, str, float):
        chi_values = {}
        for key in expected.keys():
            word_in_expected = expected.get(key)
            word_in_observed = observed.get(key)
            chi_values[key] = ((word_in_observed - word_in_expected) ** 2 / word_in_expected)
        return chi_values
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

    if correct_dict(chi_values, str, float) and isinstance(alpha, float):
        criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
        if alpha in criterion.keys():
            significant_words = {}
            for key, value in chi_values.items():
                if value > criterion[alpha]:
                    significant_words[key] = value
            return significant_words
        return None
    return None
