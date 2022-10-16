"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
import json
from pathlib import Path
from string import punctuation
from typing import Optional, Sequence, Mapping, Any
import re
from lab_1_keywords_tfidf.main import check_list

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_dict(user_input: Any, type_k: Any, type_v: Any, can_be_empty: bool) -> bool:
    """
    Checks if an object is a dictionary
    with requested keys and values types
    """
    if not isinstance(user_input, dict):
        return False
    if not user_input and can_be_empty is False:
        return False
    for key, value in user_input.items():
        if not isinstance(key, type_k) and not isinstance(value, type_v):
            return False
    return True


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(text, str) and text):
        return None
    more_punctuation = punctuation + '–—];:¡¿⟨⟩&]«»…⋯‹›“”' + '\n'
    for i in more_punctuation:
        text = text.replace(i, '!')
    text_list = text.split('!')
    words_list = [strings.strip() for strings in text_list]
    return [string for string in words_list if string != '']


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not (check_list(phrases, str, False) and check_list(stop_words, str, False)):
        return None
    no_stop_words_list = []
    for phrase in phrases:
        new_phrase = phrase.lower().split()
        to_store_phrase_list = []
        for word in new_phrase:
            if word not in stop_words:
                to_store_phrase_list.append(word)
            else:
                no_stop_words_list.append(to_store_phrase_list)
                to_store_phrase_list = []
        no_stop_words_list.append(to_store_phrase_list)
    candidate_phrases = []
    for candidate_phrase in no_stop_words_list:
        if candidate_phrase:
            candidate_phrases.append(tuple(candidate_phrase))
    return candidate_phrases
    # candidate_phrases = [tuple(candidate) for candidate in no_stop_words_list if candidate]
    # return candidate_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_list(candidate_keyword_phrases, tuple, False):
        return None
    words_list = []
    freq_dict = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            words_list.append(word)
    for word in words_list:
        freq_dict[word] = freq_dict.get(word, 0) + 1
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
    if not (check_list(candidate_keyword_phrases, tuple, False) and check_list(content_words, str, False)):
        return None
    degrees_dict = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word in content_words:
                degrees_dict[word] = degrees_dict.get(word, 0) + len(phrase)
        for word in content_words:
            if word not in degrees_dict.keys():
                degrees_dict[word] = 0
    return degrees_dict


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not (check_dict(word_degrees, str, int, False) and check_dict(word_frequencies, str, int, False)
            and word_degrees.keys() == word_frequencies.keys()):
        return None
    word_scores = {}
    for key in word_degrees.keys():
        word_scores[key] = word_degrees[key] / word_frequencies[key]
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
    if not (check_list(candidate_keyword_phrases, tuple, False) and check_dict(word_scores, str, float, False)):
        return None
    cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        value = 0
        for word in phrase:
            if word not in word_scores.keys():
                return None
            value += word_scores[word]
            cumulative_score[phrase] = value
    return cumulative_score


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
    if not (check_dict(keyword_phrases_with_scores, tuple, float, False) and isinstance(top_n, int)
            and isinstance(max_length, int) and top_n and max_length):
        return None
    if top_n <= 0 or max_length <= 0:
        return None
    sorted_list = [' '.join(phrase) for phrase, value in sorted(keyword_phrases_with_scores.items(),
                                                                key=lambda pair: pair[1], reverse=True)
                   if len(phrase) <= max_length]
    return sorted_list[:top_n]


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
    if not (check_list(candidate_keyword_phrases, tuple, False) and check_list(phrases, str, False)):
        return None
    pairs_of_phrases = []
    combined_phrases = [' '.join(kw_phrase) for kw_phrase in candidate_keyword_phrases]
    for idx, phrase1 in enumerate(combined_phrases):
        for idx2, phrase2 in enumerate(combined_phrases):
            if idx2 == idx + 1:
                pair = phrase1, phrase2
                pairs_of_phrases.append(pair)
    counter = {}
    for pair in pairs_of_phrases:
        counter[pair] = counter.get(pair, 0) + 1
    doubles = {pair: count for pair, count in counter.items() if count > 1}
    new_possible_candidates = []
    for key in doubles:
        for phrase in phrases:
            if key[0] in phrase and key[1] in phrase:
                new_possible_candidates.extend(re.findall(f'{key[0]} .* {key[1]}', phrase))
    new_candidates = []
    for phrase in new_possible_candidates:
        if new_possible_candidates.count(phrase) > 1:
            if tuple(phrase.split()) not in new_candidates:
                new_candidates.append(tuple(phrase.split()))
    return new_candidates


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
    if not (check_list(candidate_keyword_phrases, tuple, False) and check_dict(word_scores, str, float, False)
            and check_list(stop_words, str, False)):
        return None
    candidates_cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        value = 0
        for word in phrase:
            if word in stop_words:
                word_scores[word] = 0
            value += word_scores[word]
        candidates_cumulative_score[phrase] = value
    return candidates_cumulative_score


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    if not (isinstance(text, str) and text and max_length and isinstance(max_length, int)
            and not isinstance(max_length, bool) and max_length > 0):
        return None
    pass


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not isinstance(path, Path):
        return None
    with open(path, 'r', encoding='utf-8') as stop_words:
        return dict(json.load(stop_words))
