"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping
import copy

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str):
        return None
    if not text:
        return None
    level_one = ''
    for symbol in text:
        if symbol != ' ' and not symbol.isalnum():
            level_one += '.'
        else:
            level_one += symbol
    level_two = ''
    for idx, symbol in enumerate(level_one):
        if not (idx not in [-1, 0] and symbol == " " and level_one[idx + 1] == "."):
            level_two += symbol
    level_three = ''
    for idx, symbol in enumerate(level_two):
        if symbol != '.' or level_two[idx - 1] != '.':
            level_three += symbol
    level_three = level_three.replace(". ", ".")
    level_three = level_three.split('.')
    level_four = []
    for element in level_three:
        if element != "":
            level_four += [element]
    return level_four


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(phrases, list) or not isinstance(stop_words, list):
        return None
    if not all(isinstance(element, str) for element in phrases):
        return None
    if phrases == [] or stop_words == []:
        return None
    result = []
    sentences = copy.deepcopy(phrases)
    for phrase in sentences:
        phrase = phrase.lower()
        phrase = phrase.split()
        points = []
        for element in phrase:
            if element not in stop_words:
                points.append(element)
            else:
                points.append('.')
        tokens = ' '.join(points)
        back_to_string = tokens.split('.')
        dividing = []
        for element in back_to_string:
            if element != '' and element != ' ':
                dividing.append(element.strip())
        list_of_tuples = []
        for element in dividing:
            list_of_tuples.append(tuple(element.split()))
        result += list_of_tuples
    return result


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(candidate_keyword_phrases, list) or candidate_keyword_phrases == []:
        return None
    only_lists = []
    for element in candidate_keyword_phrases:
        only_lists.extend(list(element))
    freq_dict = {}
    for element in only_lists:
        freq_dict[element] = only_lists.count(element)
    return freq_dict


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
    if (not isinstance(candidate_keyword_phrases, list)
            or candidate_keyword_phrases == []
            or not isinstance(content_words, list)
            or content_words == []):
        return None
    degree_dict = {}
    for word in content_words:
        length = 0
        for phrase in candidate_keyword_phrases:
            if word in phrase:
                length += len(phrase)
        degree_dict[word] = length
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
    if (not isinstance(word_degrees, dict) or word_degrees == {}
            or not isinstance(word_frequencies, dict) or word_frequencies == {}):
        return None
    if word_degrees.keys() != word_frequencies.keys():
        return None
    scores_dict = {}
    for word in word_degrees.keys():
        scores_dict[word] = word_degrees[word]/word_frequencies[word]
    return scores_dict


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
    if (not isinstance(candidate_keyword_phrases, list) or candidate_keyword_phrases == []
            or not isinstance(word_scores, dict) or word_scores == {}):
        return None
    cum_dict = {}
    for phrase in candidate_keyword_phrases:
        phrase_score = 0
        for word in phrase:
            if word not in word_scores:
                return None
            else:
                phrase_score += word_scores[word]
        cum_dict[phrase] = phrase_score
    return cum_dict


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
    if (not isinstance(keyword_phrases_with_scores, dict)
            or keyword_phrases_with_scores == {}
            or not isinstance(top_n, int)
            or top_n <= 0
            or not isinstance(max_length, int)
            or max_length <= 0):
        return None
    only_small_phrases = {}
    for phrase, score in keyword_phrases_with_scores.items():
        if len(phrase) <= max_length:
            only_small_phrases[phrase] = score
    sorted_items = sorted(only_small_phrases.items(), key=lambda x: x[1], reverse=True)
    top_phrases = sorted_items[:top_n]
    top_simple = []
    for element in top_phrases:
        top_simple.append(' '.join(element[0]))
    return top_simple


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
