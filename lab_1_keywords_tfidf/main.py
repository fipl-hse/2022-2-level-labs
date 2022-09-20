from typing import Optional, Union
import string

def clean_and_tokenize(text: str) -> Optional[list[str]]:
    if isinstance(text, str):
        text = text.lower()
        for element in string.punctuation:
            text = text.replace(element, '')
        tokens = [element for element in text.split()]
        print('Неочищенный список слов текста:')
        print(tokens)
        return(tokens)
    else:
        return None

def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    if (isinstance(tokens, list) and tokens != [] and all(isinstance(t, str) for t in tokens)
    and isinstance(stop_words, list)):
        tokens_clean = [i for i in tokens if i not in stop_words]
        print('Чистый список слов текста:')
        print(tokens_clean)
        return(tokens_clean)
    else:
        return None

def calculate_frequencies(tokens_clean: list[str]) -> Optional[dict[str, int]]:
    if (isinstance(tokens_clean, list) and tokens_clean != [] and all(isinstance(t, str) for t in tokens_clean)):
        frequencies = {i: tokens_clean.count(i) for i in tokens_clean}
        print('Подсчёт слов в тексте:')
        print(frequencies)
        return(frequencies)
    else:
        return None

def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
    if (isinstance(frequencies, dict) and frequencies != {} and isinstance(top, int)
    and top is not (True or False) and top > 0):
        sorted_values = sorted(frequencies.values(), reverse=True)
        sorted_dict = {}
        for i in sorted_values:
            for k in frequencies.keys():
                if frequencies[k] == i:
                    sorted_dict[k] = frequencies[k]
                    break
        print('Топ 6 самых частых слов в тексте:')
        global top_six
        words = list(sorted_dict.keys())
        top_six = words[:top]
        print(top_six)
        return(top_six)
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
    pass

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
    pass

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