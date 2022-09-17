"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union, List, Dict


def clean_and_tokenize(text: str) -> Optional[List[str]]:
    """
    Removes punctuation, casts to lowercase, splits into tokens

    Parameters:
    text (str): Original text

    Returns:
    list[str]: A sequence of lowercase tokens with no punctuation

    In case of corrupt input arguments, None is returned
    """
    if isinstance(text, str):
        punctuation = '''!,()[]-{};:“'"”\<>./?@#$%^&*_~'''
        for i in punctuation:
            text = text.replace(i, '')
        text = text.lower().strip().split()
        '''если превращает в список - делает это побуквенно, поэтому нужен сплит'''
        return text
    else:
        return None


def remove_stop_words(tokens: List[str], stop_words: List[str]) -> Optional[List[str]]:
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
        no_stop_words = []
        for word in tokens:
            if word not in stop_words:
                no_stop_words.append(word)
        return no_stop_words
    else:
        return None


def calculate_frequencies(tokens: List[str]) -> Optional[Dict[str, int]]:
    """
    Composes a frequency dictionary from the token sequence

    Parameters:
    tokens (List[str]): Token sequence to count frequencies for

    Returns:
    Dict: {token: number of occurrences in the token sequence} dictionary

    In case of corrupt input arguments, None is returned
    """
    if isinstance(tokens, list):
        token_freq = {}
        for word in tokens:
            token_freq[word] = token_freq.get(word, 0) + 1
    #     d[key] = value, так добавляем в него элемент
        return token_freq
    else:
        return None


def get_top_n(frequencies: Dict[str, Union[int, float]], top: int) -> Optional[List[str]]:
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
    if isinstance(frequencies, dict) and isinstance(top, int):
        top_dic = {}
        top_values = sorted(frequencies.values(), reverse=True)
        top_n_values = top_values[:top]
        for i in top_n_values:
            for k in frequencies.keys():
                if frequencies[k] == i:
                    top_dic[k] = frequencies[k]
        return list(top_dic.keys())
    else:
        return None


def calculate_tf(frequencies: Dict[str, int]) -> Optional[Dict[str, float]]:
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
        tf_dic = {}
        values = list(frequencies.values())
        n_d = sum(values)
        for k, v in frequencies.items():
            tf_dic[k] = v/n_d
        return tf_dic
    else:
        return None


def calculate_tfidf(term_freq: Dict[str, float], idf: Dict[str, float]) -> Optional[Dict[str, float]]:
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
        tfidf_dic = {}
        for key, value in idf.items():
            for k, v in term_freq.items():
                if idf[key] != 0:
                    tfidf_dic[k] = v*value
                else:
                    tfidf_dic[k] = v*47
        return tfidf_dic
    else:
        return None


def calculate_expected_frequency(
        doc_freqs: Dict[str, int], corpus_freqs: Dict[str, float]
) -> Optional[Dict[str, float]]:
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
    # $j$ - количество вхождений слова $t$ в документ $d$
    # $k$ - количество вхождений слова $t$ во все тексты коллекции $D$
    # $l$ - количество вхождений всех слов, кроме $t$, в документ $d$
    # $m$ - количество вхождений всех слов, кроме $t$, в коллекцию документов $D$
    # $$Expected = \frac {(a + b) \times(a + c)}{a + b + c + d}$$
    if isinstance(doc_freqs, dict) and isinstance(corpus_freqs, dict):
        exp_dic = {}
        for keys, values in doc_freqs.items():
            j = doc_freqs[keys]
            k = corpus_freqs[keys]
            l = sum(list(doc_freqs.values())) - j
            m = sum(list(corpus_freqs.values())) - k
            formula = ((j + k) * (j + l)) / (j + k + l + m)
            exp_dic[keys] = formula
        return exp_dic
    else:
        return None


def calculate_chi_values(expected: Dict[str, float], observed: Dict[str, int]) -> Optional[Dict[str, float]]:
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


def extract_significant_words(chi_values: Dict[str, float], alpha: float) -> Optional[Dict[str, float]]:
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
