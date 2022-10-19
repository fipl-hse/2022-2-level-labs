"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from itertools import pairwise
from pathlib import Path
import re
from typing import Optional, Sequence, Mapping, Union, Any
from lab_1_keywords_tfidf.main import check_list, check_dict, check_positive_int



KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_types(user_var: Any, expected_type: Any, can_be_empty: bool = False) -> bool:
    """
    Checks type of variable and compares it with expected type.
    For dict and list checks whether their elements are empty (regulated with can_be_empty)
    """
    if not (isinstance(user_var, expected_type) and user_var):
        return False
    if expected_type == list:
        for element in user_var:
            if not element and can_be_empty is False:
                return False
    elif expected_type == dict:
        for key, value in user_var.items():
            if not (key and value):
                return False
    return True




def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str) or not text:
        return None
    punctuation = '.;:¡!¿?…⋯‹›«»\\"“”[]()⟨⟩}{&|-–~—'
    for i in text:
        if i in punctuation:
            text = text.replace(i, ',')
    separator_list = text.split(',')
    tokens = []
    for token in separator_list:
        token = token.strip()
        if token:
            tokens.append(token)
    return tokens



def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not (check_types(phrases, list) and check_types(stop_words, list)):
        return None
    candidate_keyword_phrases = []
    candidate_phrases_tuple = []
    for phrase in phrases:
        phrase = phrase.lower()
        words = phrase.split()
        remove_stop_words = []
        for word in words:
            if word in stop_words:
                help_keywords_candidate = tuple(remove_stop_words)
                candidate_phrases_tuple.append(help_keywords_candidate)
                remove_stop_words = []
            else:
                remove_stop_words.append(word)
        candidate_phrases_tuple.append(remove_stop_words)
    for words in candidate_phrases_tuple:
        if words:
            candidate_keyword_phrases.append(tuple(words))
    return candidate_keyword_phrases



def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_list(candidate_keyword_phrases, tuple, False):
        return None
    frequencies = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies



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
    if not check_list(candidate_keyword_phrases, tuple, False) or not  check_list(content_words, str, False):
        return None
    dict_degree = {}
    for phrase in candidate_keyword_phrases:
        for word in content_words:
            if word in phrase:
                dict_degree[word] = len(phrase) + dict_degree.get(word, 0)
            elif word not in dict_degree:
                dict_degree[word] = 0
    return dict_degree



def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not check_dict(word_degrees, str, int, False) or not check_dict(word_frequencies, str, int, False):
        return None
    dict_scores = {}
    for word in word_degrees.keys():
        if word not in word_frequencies:
            return None
        dict_scores[word] = word_degrees[word] / word_frequencies[word]
    return dict_scores


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
    if not check_list(candidate_keyword_phrases, tuple, False) or not  check_dict(word_scores, str, float, False):
        return None
    cumulative_score_dict = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score = 0
        for word in phrase:
            if word not in word_scores:
                return None
            cumulative_score += int(word_scores[word])
        cumulative_score_dict[phrase] = cumulative_score
    return cumulative_score_dict



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
    if not (check_types(keyword_phrases_with_scores, dict)
            and check_positive_int(top_n)
            and check_positive_int(max_length)):
        return None
    top_words_list = []
    sorted_keys = sorted(keyword_phrases_with_scores, reverse=True,
                         key=lambda word: keyword_phrases_with_scores[word])
    for phrase in sorted_keys:
        if len(phrase) <= max_length:
            phrase = ' '.join(phrase)
            top_words_list.append(phrase)
    return top_words_list[:top_n]




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
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_list(phrases, str, False):
        return None
    joining_phrases = []
    for key_word in candidate_keyword_phrases:
        phrase = ' '.join(key_word)
        joining_phrases.append(phrase)
    list_of_pairs = list(pairwise(joining_phrases))
    frequencies = {}
    for item in list_of_pairs:
        frequencies[item] = frequencies.get(item, 0) + 1
    stop_words_list = []
    for key_word in frequencies:
        if frequencies[key_word] > 2:
            continue
        for phrase in phrases:
            if (key_word[0] and key_word[1]) in phrase:
                stop_words_list.extend(re.findall(f'{key_word[0]}.*{key_word[1]}', phrase))
    new_keywords_phrases = []
    for keywords_with_stop_words in dict.fromkeys(stop_words_list):
        if stop_words_list.count(keywords_with_stop_words) > 1:
            new_keywords_phrases.append(tuple(keywords_with_stop_words.lower().split()))
    return new_keywords_phrases


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
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_list(stop_words, str, False) \
            or not check_dict(word_scores, str, Union[int, float], False):
        return None
    candidates_cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        candidates_cumulative_score[phrase] = 0
        for word in phrase:
            if word not in stop_words:
                candidates_cumulative_score[phrase] += int(word_scores[word])
    return candidates_cumulative_score



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
