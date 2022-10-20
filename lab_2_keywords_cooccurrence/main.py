"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping
from lab_1_keywords_tfidf.main import check_list, check_dict

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str) or len(text) == 0:
        return None

    punct = '''.,;:¡!¿?…⋯‹›«»\\"“”\[\]()⟨⟩}{&]|[-–~—]'''
    for symbol in punct:
        text = text.replace(symbol, ',')
        # проходится по всем знакам пункуации и заменяет все на запятые
    coma_split = text.split(',')
    # разделяет по запятым на элементы, далее работа со списком
    final_list = []
    for i in coma_split:
        i = i.strip()
        # Метод strip() возвращает копию строки, удаляя как начальные, так и конечные символы
        # (в зависимости от переданного строкового аргумента).
        # Метод удаляет символы как слева, так и справа в зависимости от аргумента
        # (строка, определяющая набор символов, которые необходимо удалить).
        if i:
            final_list.append(i)
        # проверка пустой ли i или нет, потому что если там были пробелы,
        # то пердыдущее действия их все удалило и он пуст, нам такой мусор не нужен
    return final_list


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not check_list(phrases, str, False) or not check_list(stop_words, str, False):
        return None
    final_list = []
    final_list_tuples = []
    for phrase in phrases:
        phrase = phrase.lower().split()
        for word in phrase:
            if word in stop_words:
                index = phrase.index(word)
                phrase[index] = ','
        phrase = " ".join(phrase)
        phrase = phrase.split(',')
        # разделяет по запятым на элементы, далее работа со списком
        for i in phrase:
            i = i.strip(',')
            # Метод strip() возвращает копию строки, удаляя как начальные, так и конечные символы
            # (в зависимости от переданного строкового аргумента).
            # Метод удаляет символы как слева, так и справа в зависимости от аргумента
            # (строка, определяющая набор символов, которые необходимо удалить).
            if i:
                i = i.strip()
                final_list.append(i)
            # проверка пустой ли i или нет, потому что если там были пробелы,
            # то пердыдущее действия их все удалило и он пуст, нам такой мусор не нужен

    for string in final_list:
        if string:
            final_list_tuples.append(tuple(string.split(' ')))

    return final_list_tuples

def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    tokens = []
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            tokens.append(word)
    if tokens:
        my_dict = {token: tokens.count(token) for token in tokens}
    return my_dict


def calculate_word_degrees(candidate_keyword_phrases: KeyPhrases,
                           content_words: Sequence[str]) -> Optional[Mapping[str, int]]:
    """
    Calculates the word degrees based on the candidate keyword phrases list
    Degree of a word is equal to the total length of all keyword phrases the word is found in

    :param content_words: the content words from the candidate keywords
    :param candidate_keyword_phrases: the candidate keyword phrases for the text
    :return: the words and their degrees

    In case of corrupt input arguments, None is returned
    """
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_list(content_words, str, False):
        return None
    if len(candidate_keyword_phrases) == 0 or len(content_words) == 0:
        return None
    word_degrees = {}
    word_degrees_content = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word in word_degrees:
                for words in phrase:
                    word_degrees[word] += [words]
            else:
                word_degrees[word] = []
                for words in phrase:
                    word_degrees[word] += [words]
    for word in content_words:
        if word not in word_degrees:
            word_degrees_content[word] = 0
        else:
            count = len(word_degrees[word])
            word_degrees_content[word] = count
    return word_degrees_content

def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not (check_dict(word_degrees, str, int, False) and check_dict(word_frequencies, str, int, True)):
        return None
    if len(word_degrees) == 0 or len(word_frequencies) == 0:
        return None
    word_scores = {}
    for word in word_degrees.keys():
        if word not in word_frequencies.keys():
            return None
        word_scores[word] = word_degrees[word] / word_frequencies[word]
    return word_scores

def calculate_cumulative_score_for_candidates(candidate_keyword_phrases: KeyPhrases,
                                              word_scores: Mapping[str, float]) -> Optional[Mapping[KeyPhrase, float]]:
    """
    Calculate cumulative score for each candidate keyword phrase. Cumulative score for a keyword phrase equals to
    the sum of the word scores of each keyword phrase's constituent

    :param candidate_keyword_phrases: a list of candidate keyword phrases
    :param word_scores: word scores
    :return: a dictionary containing the mapping between the candidate keyword phrases and respective cumulative scores

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    if not isinstance(word_scores, dict) or not word_scores:
        return None
    if len(candidate_keyword_phrases) == 0 or len(word_scores) == 0:
        return None
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word not in word_scores:
                return None
    cumulative_score_for_candidates = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score_for_candidates[phrase] = 0.0
        for word in phrase:
            cumulative_score_for_candidates[phrase] += word_scores[word]
    return cumulative_score_for_candidates

def get_top_n(keyword_phrases_with_scores: Mapping[KeyPhrase, float],
              top_n: int,
              max_length: int) -> Optional[Sequence[str]]:
    """
    Extracts the top N keyword phrases based on their scores and lengths

    :param keyword_phrases_with_scores: a dictionary containing the keyword phrases and their cumulative scores
    :param top_n: the number of the keyword phrases to extract
    :param max_length: maximal length of a keyword phrase to be considered
    :return: a list of keyword phrases sorted by their scores in descending order

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(keyword_phrases_with_scores, dict) or not keyword_phrases_with_scores:
        return None
    if not isinstance(top_n, int) or not top_n:
        return None
    if not isinstance(max_length, int) or not max_length:
        return None
    if len(keyword_phrases_with_scores) == 0 or top_n <= 0 or max_length <= 0:
        return None
    top_n_dict = {}
    top_n_list = []
    top_n_strings_list = []
    for key, value in keyword_phrases_with_scores.items():
        if len(key) <= max_length:
            top_n_dict[key] = value
    top_n_list = [key for (key, value) in sorted(top_n_dict.items(), key=lambda x: x[1], reverse=True)][:top_n]
    for keywords_tuple in top_n_list:
        keywords_tuple = list(keywords_tuple)
        top_n_strings_list.append(' '.join(keywords_tuple))
    return top_n_strings_list

def extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases: KeyPhrases,
                                                     phrases: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Extracts the adjoining keyword phrases from the candidate keywords Sequence and
    builds new candidate keywords containing stop words

    Adjoining keywords: such pairs that are found at least twice in the candidate keyword phrases list one after another

    To build a new keyword phrase the following is required:
        1. Find the first constituent of the adjoining keyword phrase in the phrases followed by:
            a stop word and the second constituent of the adjoining keyword phrase
        2. Combine these three pieces in the new candidate keyword phrase, i.e.:
            new_candidate_keyword = [first_constituent, stop_word, second_constituent]

    :param candidate_keyword_phrases: a list of candidate keyword phrases
    :param phrases: a list of phrases
    :return: a list containing the pairs of candidate keyword phrases that are found at least twice together

    In case of corrupt input arguments, None is returned
    """
    pass


def calculate_cumulative_score_for_candidates_with_stop_words(candidate_keyword_phrases: KeyPhrases,
                                                              word_scores: Mapping[str, float],
                                                              stop_words: Sequence[str]) \
        -> Optional[Mapping[KeyPhrase, float]]:
    """
    Calculate cumulative score for each candidate keyword phrase. Cumulative score for a keyword phrase equals to
    the sum of the word scores of each keyword phrase's constituent except for the stop words

    :param candidate_keyword_phrases: a list of candidate keyword phrases
    :param word_scores: word scores
    :param stop_words: a list of stop words
    :return: a dictionary containing the mapping between the candidate keyword phrases and respective cumulative scores

    In case of corrupt input arguments, None is returned
    """
    pass


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    pass


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    pass
