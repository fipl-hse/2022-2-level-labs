from typing import Optional, Union

with open("Дюймовочка.txt", "r", encoding="utf-8") as f1:
    f2 = open("stop_words.txt", "r", encoding="utf-8")
    text = f1.read()
    stopwords = f2.read()
    stop_words = stopwords.split()
    tokens = []
    tokens_clean = []
    frequencies = {}
    top = 6

def clean_and_tokenize(text: str):
    global tokens
    punctuation = '''!.?,:;"'-()'''
    text = text.lower()
    for element in text:
        if element in punctuation:
            text = text.replace(element,' ')
    tokens = [element for element in text.split()]
    print('Неочищенный список слов текста:')
    print(tokens)
    return(tokens)

clean_and_tokenize(text)

def remove_stop_words(tokens: list[str], stop_words: list[str]):
    global tokens_clean
    tokens_clean = [i for i in tokens if i not in stop_words]
    print('Чистый список слов текста:')
    print(tokens_clean)
    return(tokens_clean)

remove_stop_words(tokens,stop_words)


def calculate_frequencies(tokens_clean: list[str]):
    global frequencies
    frequencies = {i: tokens_clean.count(i) for i in tokens_clean}
    print('Подсчёт слов в тексте:')
    print(frequencies)
    return(frequencies)

calculate_frequencies(tokens_clean)


def get_top_n(frequencies: dict[str,int], top: int):
    sorted_values = sorted(frequencies.values(), reverse=True)
    sorted_dict = {}
    for i in sorted_values:
        for k in frequencies.keys():
            if frequencies[k] == i:
                sorted_dict[k] = frequencies[k]
                break
    print('Топ 6 самых частых слов в тексте:')
    for i in range(top):
        print(list(sorted_dict.items())[i])

get_top_n(frequencies, top)


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
