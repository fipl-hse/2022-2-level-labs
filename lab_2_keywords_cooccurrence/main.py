"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from itertools import pairwise
from pathlib import Path
from typing import Any, Optional, Sequence, Mapping

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_sequence(obj: Any, element_type: type, can_be_empty: bool) -> bool:
    """
    Checks if the object is a list containing elements of a certain type

    :param obj: an object that is expected to be a list
    :param element_type: the expected type of list elements
    :param can_be_empty: True if the object can be empty, False otherwise
    :return: bool - True if the object is list containing elements of the given type, False otherwise
    """
    if (can_be_empty is False and obj) or (can_be_empty is True and not obj):
        if isinstance(obj, (list, tuple)):
            for element in obj:
                if isinstance(element, element_type):
                    return True
    return False


def check_mapping(obj: Any, key_type: type, val_type: type, can_be_empty: bool) -> bool:
    """
    Checks if the object is a dictionary containing keys and values of a certain type

    :param obj: an object that is expected to be a dictionary
    :param key_type: the expected type of dictionary keys
    :param val_type: the expected type of dictionary values
    :param can_be_empty: True if the object can be empty, False otherwise
    :return: bool - True if the object is a dictionary containing keys and values of the given type, False otherwise
    """
    if (can_be_empty is False and obj) or (can_be_empty is True and not obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(key, key_type) and isinstance(val, val_type):
                    return True
    return False


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not(isinstance(text, str) and text):
        return None
    for punc in r""".¡!¿?"“”…⋯#$%&'()*+-/:;<=>‹›⟨⟩@[\]^_`{|}-–~—«»""":
        text = text.replace(punc, ',')
    phrases = [phrase.strip() for phrase in text.split(',') if phrase.strip()]
    return phrases


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not(check_sequence(phrases, str, False) and not isinstance(phrases, str)
           and check_sequence(stop_words, str, False) and not isinstance(stop_words, str)):
        return None

    splt_phrases = [phrase.lower().split() for phrase in phrases]
    candidate_keyword_phrases = []
    keyword_phrase = []

    for phrase_idx, splt_phrase in enumerate(splt_phrases):
        for word_idx, word in enumerate(splt_phrase):

            if word not in stop_words:
                if word_idx == 0 and keyword_phrase:
                    candidate_keyword_phrases.append(tuple(keyword_phrase))
                    keyword_phrase = []

                keyword_phrase.append(word)

                if word_idx == len(splt_phrase) - 1 and phrase_idx == len(splt_phrases) - 1:
                    candidate_keyword_phrases.append(tuple(keyword_phrase))

            elif word in stop_words and keyword_phrase:
                candidate_keyword_phrases.append(tuple(keyword_phrase))
                keyword_phrase = []

    return candidate_keyword_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_sequence(candidate_keyword_phrases, tuple, False):
        return None
    keyword_freq = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            keyword_freq[word] = keyword_freq.get(word, 0) + 1
    return keyword_freq


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
    if not(check_sequence(candidate_keyword_phrases, tuple, False) and check_sequence(content_words, str, False)):
        return None

    degree_dict = dict.fromkeys(content_words, 0)
    for word in content_words:
        for phrase in candidate_keyword_phrases:
            if word in phrase:
                degree_dict[word] = degree_dict.get(word, 0) + len(phrase)

    return degree_dict


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not(check_mapping(word_degrees, str, int, False) and check_mapping(word_frequencies, str, int, False)
           and word_degrees.keys() == word_frequencies.keys()):
        return None

    return {word1: (degree / word_frequencies.get(word1, 1)) for word1, degree in word_degrees.items()}


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
    if not(check_sequence(candidate_keyword_phrases, tuple, False) and check_mapping(word_scores, str, float, False)
           and all(word_scores.get(word) for phrase in candidate_keyword_phrases for word in phrase)):
        return None

    cmltv_score_dict = {}

    for word, score in word_scores.items():
        for phrase in set(candidate_keyword_phrases):

            if word in phrase:
                cmltv_score_dict[phrase] = cmltv_score_dict.get(phrase, 0.0) + score

    return cmltv_score_dict


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
    if not(check_mapping(keyword_phrases_with_scores, tuple, float, False) and isinstance(top_n, int) and top_n > 0
           and isinstance(max_length, int) and max_length > 0):
        return None

    score_sorted_phrases = sorted(keyword_phrases_with_scores.keys(), key=lambda key: keyword_phrases_with_scores[key],
                                  reverse=True)
    return [' '.join(phrase) for phrase in score_sorted_phrases if len(phrase) <= max_length][:top_n]


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
    if not(check_sequence(candidate_keyword_phrases, tuple, False) and check_sequence(phrases, str, False)):
        return None
    pairs = list(pairwise(candidate_keyword_phrases))
    phrases = [phrase.lower() for phrase in phrases]
    new_phrases = []

    for pair in pairs:
        possible_new1 = list(pair[0])
        possible_new2 = list(pair[1])
        phrase_freq = candidate_keyword_phrases.count(pair[0])
        next_phrase_freq = candidate_keyword_phrases.count(pair[1])

        for phrase in phrases:
            if (' '.join(possible_new1) in phrase and ' '.join(possible_new2) in phrase and
                    phrase_freq > 1 and next_phrase_freq > 1):
                try:
                    splt_phrase = phrase.split()
                    p1_end_idx = splt_phrase.index(possible_new1[-1])
                    p2_begin_idx = splt_phrase.index(possible_new2[0])

                    if p2_begin_idx - p1_end_idx == 2:
                        stop_word = splt_phrase[p1_end_idx + 1]
                        all_words = [word for phrase in phrases for word in phrase.split()]
                        stop_word_freq = all_words.count(stop_word)

                        possible_phrase = possible_new1 + [stop_word] + possible_new2

                        if (possible_phrase not in new_phrases and
                                (stop_word_freq >= phrase_freq or stop_word_freq >= next_phrase_freq)):
                            new_phrases.append(tuple(possible_phrase))

                except ValueError:
                    pass

    return list(set(new_phrases))


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
    if not(check_sequence(candidate_keyword_phrases, tuple, False) and check_mapping(word_scores, str, float, False)
           and check_sequence(stop_words, str, False) and not isinstance(stop_words, str)):
        return None

    cmltv_score_dict_wtih_stops = {}

    for word, score in word_scores.items():
        for phrase in set(candidate_keyword_phrases):

            if word in phrase and word not in stop_words:
                cmltv_score_dict_wtih_stops[phrase] = cmltv_score_dict_wtih_stops.get(phrase, 0.0) + score

    return cmltv_score_dict_wtih_stops


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
