"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping
import re
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
    delimiters_list = (',', '!', '?', '\n', ':', ';', ' – ', '¡', '¿', '…', '⋯', '‹', '›', '«', '»', '[', ']',
                        '(', ')', '⟨', '⟩', '}', '{', '&', '|', '-', '–', '~', '—', '\\\\"“”\\', '\\')
    for item in delimiters_list:
        text = text.replace(item, '.')
    phrases_list = text.split('.')
    not_empty_phrases_list = [phrase.strip() for phrase in phrases_list if phrase and phrase != ' ']
    return not_empty_phrases_list


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    #что делать с Sequence
    if not isinstance(phrases, list):
        return None
    for item in phrases:
        if not isinstance(item, str):
            return None
    if not isinstance(stop_words, list):
        return None
    if not stop_words or not phrases:
        return None
    for item1 in stop_words:
        if not isinstance(item1, str):
            return None

    key_phrases = []
    for phrase in phrases:
        phrase = phrase.lower()
        phrase_word_list = phrase.split()
        for element in phrase_word_list:
            if element in stop_words:
                #phrase = phrase.replace(f' {element} ', ',')
                if phrase_word_list[0] == element:
                    phrase = phrase.replace(f'{element} ', ' , ')
                if phrase_word_list[-1] == element:
                    phrase = phrase.replace(f' {element}', ' , ')
                else:
                    phrase = phrase.replace(f' {element} ', ' , ')

        phrase = " ".join(phrase.split())
        keyword_phrase = phrase.split(',')
        keyword_phrase1 = []
        for element1 in keyword_phrase:
            if element1 and element1 != ' ':
                keyword_phrase1.append(element1)
        key_phrases.append(keyword_phrase1)

    probably_keywords = []
    for lst in key_phrases:
        for words_str in lst:
            words_tuple = tuple(words_str.split())
            probably_keywords.append(words_tuple)
    return probably_keywords


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(candidate_keyword_phrases, list):
        return None
    for item in candidate_keyword_phrases:
        if not isinstance(item, tuple):
            return None
        for word in item:
            if not isinstance(word, str):
                return None
    if not candidate_keyword_phrases:
        return None
    frequencies_dict = {}
    word_list = []
    for tpl in candidate_keyword_phrases:
        for element in tpl:
            word_list.append(element)
    unique_words = set(word_list)

    for phrase in unique_words:
        count = word_list.count(phrase)
        frequencies_dict[phrase] = count
    return frequencies_dict
    pass


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
    if not isinstance(candidate_keyword_phrases, list):
        return None
    for item in candidate_keyword_phrases:
        if not isinstance(item, tuple):
            return None
        for word in item:
            if not isinstance(word, str):
                return None
    if not candidate_keyword_phrases:
        return None
    if not isinstance(content_words, list):
        return None
    for item in content_words:
        if not isinstance(item, str):
            return None
    if not content_words:
        return None
    word_degree_dict = {}
    word_degree = 0
    for word in content_words:
        for phrase in candidate_keyword_phrases:
            if word in phrase:
                word_degree += len(phrase)
        word_degree_dict[word] = word_degree
        word_degree = 0
    return word_degree_dict
    pass


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(word_degrees, dict):
        return None
    for key, value in word_degrees.items():
        if not (isinstance(key, str) and isinstance(value, int)):
            return None
    if not word_degrees:
        return None
    if not isinstance(word_frequencies, dict):
        return None
    for key, value in word_frequencies.items():
        if not (isinstance(key, str) and isinstance(value, int)):
            return None
    if not word_frequencies:
        return None

    word_score_dict = {}
    for key, value in word_degrees.items():
        for key1, value1 in word_frequencies.items():
            if key == key1:
                word_score = int(value)/int(value1)
            if key not in word_frequencies.keys():
                return None

        word_score_dict[key] = word_score
    return word_score_dict


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
    if not isinstance(word_scores, dict):
        return None
    if not isinstance(candidate_keyword_phrases, list):
        return None
    if not candidate_keyword_phrases:
        return None
    if not word_scores:
        return None
    for key, value in word_scores.items():
        if not (isinstance(key, str) and isinstance(value, float)):
            return None
    for item in candidate_keyword_phrases:
        if not isinstance(item, tuple):
            return None
        for word in item:
            if not isinstance(word, str):
                return None
    cumulative_score_for_candidates_dict = {}
    cumulative_score = 0
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word in word_scores.keys():
                cumulative_score += int(word_scores[word])
            else:
                return None
        cumulative_score_for_candidates_dict[phrase] = cumulative_score
        cumulative_score = 0
    return cumulative_score_for_candidates_dict
    pass


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
    if not keyword_phrases_with_scores:
        return None
    if not isinstance(keyword_phrases_with_scores, dict):
        return None
    for key, value in keyword_phrases_with_scores.items():
        if not isinstance(key, tuple) or not isinstance(value, int):
            return None
        for item in key:
            if not isinstance(item, str):
                return None
    if not isinstance(top_n, int):
        return None
    if not isinstance(max_length, int):
        return None
    keyword_phrases_with_scores_limited = {}
    for phrase, score in keyword_phrases_with_scores.items():
        if len(phrase) <= max_length:
            phrase = ' '.join(phrase)
            keyword_phrases_with_scores_limited[phrase] = score
    if not keyword_phrases_with_scores_limited:
        return None

    if top_n > len(keyword_phrases_with_scores_limited.values()):
        sorted_values = sorted(keyword_phrases_with_scores_limited.values(), reverse=True)
    else:
        sorted_values = sorted(keyword_phrases_with_scores_limited.values(), reverse=True)[:top_n]
    sorted_dict = {}

    for item in sorted_values:
        for key in keyword_phrases_with_scores_limited.keys():
            if keyword_phrases_with_scores_limited[key] == item:
                sorted_dict[key] = keyword_phrases_with_scores_limited[key]

    return sorted_dict


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
