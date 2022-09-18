"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union


def clean_and_tokenize(text: str):
    if isinstance(text, str) is False:
        return None
    else:
        text = text.lower()
        text = text.strip()
        marks = ",.:;!?—(){}[]&"
        # добавить удаление  пробелов рядом с пробелами
        for i in text:
            if i in marks:
                text = text.replace(i, "")
            else:
                pass
        list = text.split()
        print(list)

def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
    if isinstance(tokens, list) is False or isinstance(stop_words, list) is False:
        return None
    elif len(tokens) == 0 or len(stop_words) == 0:
        return None
    for word in stop_words:
        while word in tokens:
            tokens.remove(word)   #апдейт кажется я починила но все равно нужно чекнуть (код пропускал некоторые слова)
    print(tokens)


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    frequencies = {}
    if isinstance(tokens, list) is False or len(tokens) == 0:
        return None
    for i in tokens:
        if isinstance(i, str) is False:
            return None
    else:
        for i in tokens:
            x = tokens.count(i)
            frequencies[i] = x
    print(frequencies)


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
    if isinstance(frequencies, dict) is False or (isinstance(top, (int, float)) is False ):
        return None
    else:
        freq_val = list(frequencies.values())
        words_num = 0
        for value in freq_val:
            words_num += value
        print(words_num)
        if top <= words_num:
            top_list = sorted(frequencies, key = frequencies.get, reverse = True)[:top]  #get возвращает значение по указанному ключу
        else:
            top_list = sorted(frequencies, key = frequencies.get, reverse = True)
        print(top_list)


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
    if isinstance(frequencies, dict) is False:   #добавить проверку на то, являются ли ключи строками а значения интами
        return None
    else:
        freq_values = list(frequencies.values())
        words_num = 0
        for value in freq_values:
            words_num += value
        tf_dict = {}
        for key, val in frequencies.items():
            tf = val / words_num
            tf_dict[key] = round(tf, 3)   #нужен ли тут раунд?
        print(tf_dict)


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
    pass
