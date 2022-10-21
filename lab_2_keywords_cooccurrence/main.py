"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Union
import json
from itertools import pairwise
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
    if not isinstance(text, str) or not text:
        return None
    punctuation_marks = '''.,;':¡!¿?…⋯‹›«»\\/"“”[]()⟨⟩}{&|-–~—'''
    for mark in punctuation_marks:
        text = text.replace(mark, ',')
    split_text = text.split(',')
    return [phrase1 for phrase in split_text if (phrase1 := phrase.strip())]


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
    candidates_list = []
    for phrase in phrases:
        ready_phrase = phrase.lower().split()
        candidate1 = []
        for word in ready_phrase:
            if word in stop_words:
                if candidate1:
                    candidates_list.append(tuple(candidate1))
                    candidate1.clear()
            elif word == ready_phrase[len(ready_phrase) - 1]:
                candidate1.append(word)
                candidates_list.append(tuple(candidate1))
                candidate1.clear()
            else:
                candidate1.append(word)
    return candidates_list


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_list(candidate_keyword_phrases, tuple, False):
        return None
    frequencies_for_content_words = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            frequencies_for_content_words[word] = frequencies_for_content_words.get(word, 0) + 1
    return frequencies_for_content_words


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
    word_degrees = {}
    for phrase in candidate_keyword_phrases:
        for word in content_words:
            if word in phrase:
                word_degrees[word] = word_degrees.get(word, 0) + len(phrase)
            elif word not in word_degrees:
                word_degrees[word] = 0
    return word_degrees


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
    for word in word_degrees:
        if word not in word_frequencies:
            return None
    return {word: word_degrees[word] / word_frequencies[word] for word in word_degrees if word in word_frequencies}


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
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_dict(word_scores, str, float, False):
        return None
    cumulative_score_for_candidates = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score = 0
        for word in phrase:
            if word not in word_scores:
                return None
            cumulative_score += int(word_scores[word])
        cumulative_score_for_candidates[phrase] = cumulative_score
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
    if not check_dict(keyword_phrases_with_scores, tuple, float, False) or not isinstance(top_n, int) \
            or not isinstance(max_length, int) or not max_length > 0 or not top_n > 0:
        return None
    top_phrases = sorted(keyword_phrases_with_scores.keys(), key=lambda word: keyword_phrases_with_scores[word],
                         reverse=True)
    return [" ".join(phrase) for phrase in top_phrases if len(phrase) <= max_length][:top_n]


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
    list_with_all_pairs = []
    for value1, value2 in pairwise(candidate_keyword_phrases):
        help_tuple = tuple([' '.join(value1), ' '.join(value2)])
        list_with_all_pairs.append(help_tuple)
    duplicates1 = [value for index, value in enumerate(list_with_all_pairs) if value in list_with_all_pairs[:index]]
    duplicates2 = set(duplicates1)
    key_phrases = []
    for item1 in phrases:
        for elem in duplicates2:
            first_tuple, second_tuple = elem[0], elem[1]
            if first_tuple not in item1:
                continue
            first_place, second_place = item1.find(first_tuple), item1.rfind(second_tuple[len(second_tuple) - 1])
            key_phrases.append(item1[first_place:second_place + 1])
    if not key_phrases:
        return []
    duplicates3 = [value for index, value in enumerate(key_phrases) if value in key_phrases[:index]]
    return [tuple(item2.split()) for item2 in duplicates3]


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
    result_dict = {}
    for item in candidate_keyword_phrases:
        count = 0
        for elem in item:
            for key, value in word_scores.items():
                if elem != key:
                    continue
                count += int(value)
                for stop_word in stop_words:
                    if elem == stop_word:
                        count -= int(value)
        result_dict[item] = count
    return result_dict


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    if not isinstance(text, str) or not isinstance(max_length, int) or max_length < 0 or not text:
        return None
    punctuation = '''.,;':¡!¿?…⋯‹›«»\\/"“”[]()⟨⟩}{&|-–~—'''
    clean_text = ''
    clean_text_1 = [clean_text + word for word in text.lower().replace(',', '').split() if word not in punctuation]
    frequencies = {token: clean_text_1.count(token) for token in clean_text_1}
    freq_list = sorted(frequencies.values())
    percentile = int((80 / 100) * len(freq_list))
    return [key for key, value in frequencies.items() if percentile <= value and len(key) <= max_length]


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
