"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import math

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
        text = text.lower()
        text = text.strip()
        marks = ",.:;!?—-(){}[]&"
        # добавить удаление  пробелов рядом с пробелами
        for i in text:
            if i in marks:
                text = text.replace(i, "")
        words_list = text.split()
        return (words_list)

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
    if not isinstance(tokens, list)  or not isinstance(stop_words, list):
        return None
    elif not tokens or not stop_words:
        return None
    else:
        for word in stop_words:
            while word in tokens:
                tokens.remove(word)   # апдейт кажется я починила но все равно нужно чекнуть (код пропускал некоторые слова)
        return(tokens)


def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    frequencies = {}
    if not isinstance(tokens, list) or not tokens:
        return None
    for i in tokens:
        if not isinstance(i, str):
            return None
    else:
        for i in tokens:
            x = tokens.count(i)
            frequencies[i] = x
    return (frequencies)


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
    # изменить функцию, потому что на 7 шаге она не работает
    if not isinstance(frequencies, dict) or not isinstance(top, (int, float)):
        return None
    else:
        freq_val = list(frequencies.values())
        words_num = 0
        for value in freq_val:
            words_num += value
        print(words_num)
        if top <= words_num:    # исправить в соответветствии с комментарием ментора
            """следующая абракадабра сортирует значения функций по убыванию(reverse = true, 
             если этого не писать будет по возрастанию)
             и выдает лист ключей, которым эти значения принадлежат"""
            # нужно лучше разобраться в функции sorted
            # get возвращает значение по указанному ключу
            top_list = sorted(frequencies, key=frequencies.get, reverse=True)[:top]
        else:
            top_list = sorted(frequencies, key=frequencies.get, reverse=True)
        return (top_list)


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
    if not isinstance(frequencies, dict):
        # добавить проверку на то, являются ли ключи строками а значения интами
        # апдейт - сделано
        return None
    for k, v in frequencies.items():
        if not isinstance(k, str) or not isinstance(v, int):
            return None
    else:
        freq_values = list(frequencies.values())
        words_num = sum(freq_values)
        tf_dict = {}
        for key, val in frequencies.items():
            tf = val / words_num
            tf_dict[key] = tf
        return (tf_dict)


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
    # я понятия не имею работает ли эта функция
    if not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None
    else:
        for k, v in term_freq.items():
            if not isinstance(k, str) or not isinstance(v, float):
                return None
        for k, v in term_freq.items():
            if not isinstance(k, str) or not isinstance(v, float):
                return None
        tfidf_dict = {}
        for key, v in term_freq.items():
            if key in idf:
                tfidf_dict[key] = v * idf[key]
                """в словарь со значениями tfidf добавляю слово, которое является ключом, и его значение,
                равное произведению значений данных на входе словарей(если вдруг я забуду что тут происходит )"""
            else:
                key_idf = math.log(47 / (0 + 1))
                tfidf_dict[key] = v * key_idf
            return (tfidf_dict)

def calculate_expected_frequency(
    doc_freqs: dict[str, int], corpus_freqs: dict[str, int]
) -> Optional[dict[str, float]]:
    # я не знаю, работает ли эта функция
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
    if not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict):
        return None
    else:
        for k, v in doc_freqs.items():
            if not isinstance(k, str) or not isinstance(v, int):
                return None
        for k, v in corpus_freqs.items():
            if not isinstance(k, str) or not isinstance(v, int):
                return None
        # подумать как решить это по другому
        doc_freqs_val = list(doc_freqs.values())
        col_freqs_val = list(corpus_freqs.values())
        words_in_doc = sum(doc_freqs_val)
        words_in_col =  sum(col_freqs_val)
        exp_freqs = {}
        for key, val in corpus_freqs.items():
            for i in range(doc_freqs_val):
                l = words_in_doc - doc_freqs_val[i]
                # l -  количество вхождений всех слов, кроме , в документ d
                m = words_in_col - col_freqs_val[i]
                # m - количество вхождений всех слов, кроме , в коллекцию документов D
                j = doc_freqs_val[i]
                # j - количество вхождений слова t  в документ d
                k = col_freqs_val[i]
                # k - количество вхождений слова t во все тексты коллекции D
                exp = ((j + k) * (j * l))/ (j + k + l + m)
                exp_freqs[key] = exp
        return (exp_freqs)
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
