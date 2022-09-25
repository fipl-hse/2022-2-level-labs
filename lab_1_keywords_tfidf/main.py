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
    if type(text) == str:
        target_text = text.lower().strip()
        symbols = """!()-[]{};?@#$%:'"\,./^&;*_><"""
        for symbol in target_text:
            if symbol in symbols:
                target_text = target_text.replace(symbol, '')
        target_text = target_text.split()
        tokens = target_text
        for word in tokens:
            if word == "' '":
                tokens.remove(word)
        return tokens
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
    if type(tokens) != list or type(stop_words) != list:
        return None
    else:
        new_tokens = []
        for word in tokens:
            if type(word) is not str:
                return None
            if word not in stop_words:
                new_tokens.append(word)
        for word in stop_words:
            if type(word) is not str:
                return None
    tokens = new_tokens
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
    if type(tokens) is not list:
        return None
    else:
        frequencies = {}
        for word in tokens:
            if type(word) is not str or word == "":
                return None
            else:
                frequencies[word] = tokens.count(word)
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
    if type(top) is not int or top <= 0:
        return None
    if type(frequencies) is dict:
        if frequencies == {}:
            return None
        for key in frequencies.keys():
            if type(key) is not str:
                return None
        for value in frequencies.values():
            if type(value) is not int or not float:
                return None
        if top >= len(frequencies):
            most_common = sorted(frequencies, key=frequencies.get, reverse=True)
            return most_common
        if top < len(frequencies):
            most_common = sorted(frequencies, key=frequencies.get, reverse=True)
            most_common = list(most_common[:top])
            return most_common
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
    if type(frequencies) is dict:
        for key in frequencies.keys():
            if type(key) is not str:
                return None
        lenght = 0
        for value in frequencies.values():
            if type(value) is not int:
                return None
            lenght += value
        term_freq = {}
        for word in frequencies.keys():
            freq = frequencies[word]/lenght
            term_freq[word] = freq
        return term_freq
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
    import math
    if type(term_freq) is dict and type(idf) is dict and term_freq != {}:
        for key in term_freq.keys():
            if type(key) is not str:
                return None
        for value in term_freq.values():
            if type(value) is not float:
                return None
        for key in idf.keys():
            if type(key) is not str:
                return None
        for value in idf.values():
            if type(value) is not float:
                return None
        else:
            tfidf = {}
            for word in idf.keys():
                if word not in idf.keys():
                    idf_meaning = math.log(47)
                    idf[word] = idf_meaning
            if idf == {}:
                for word, word_freq in term_freq.items():
                    tfidf[word] = word_freq * math.log(47)
            for word in idf.values():
                if word == 0:
                    max_idf = math.log(47)
                    tfidf = {term: term_freq * idf.get(term, max_idf) for term, term_freq in term_freq.items()}
                    return tfidf
            return tfidf
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
