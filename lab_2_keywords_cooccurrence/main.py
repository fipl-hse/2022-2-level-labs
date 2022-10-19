"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
from itertools import pairwise
import json
from lab_1_keywords_tfidf.main import check_positive_int, clean_and_tokenize, calculate_frequencies

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_object(user_input: Any, object_type: Any, token_type: Any = None) -> bool:
    """
    Universal check for object's type and its emptiness.
    """
    if not user_input:
        return False
    if not isinstance(user_input, object_type):
        return False
    if token_type:
        for token in user_input:
            if not isinstance(token, token_type):
                return False
    return True


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not check_object(text, str):
        return None
    punctuation_str = ',;:¡!¿?…⋯‹›«»\\\\"“”\\[\\]()⟨⟩}{&]|[-–~—]'
    clean_text = ''
    for symbol in text:
        if symbol in punctuation_str:
            clean_text += '.'
        else:
            clean_text += symbol
    text_list = clean_text.split('.')
    phrases = []
    for phrase in text_list:
        phrase = phrase.strip()
        if phrase:
            phrases.append(phrase)
    return phrases


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not check_object(phrases, list) or not check_object(stop_words, list):
        return None
    candidate_keyword_phrases = []
    for phrase in phrases:
        phrase_list = phrase.lower().split()
        for index, word in enumerate(phrase_list):
            if word in stop_words:
                phrase_list[index] = '.'
        no_stop_words = extract_phrases(' '.join(phrase_list))
        for no_stop_words_phrase in no_stop_words:
            candidate_phrase = no_stop_words_phrase.split()
            if candidate_phrase:
                candidate_keyword_phrases.append(tuple(candidate_phrase))
    return candidate_keyword_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_object(candidate_keyword_phrases, list, tuple):
        return None
    frequencies = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word not in frequencies.keys():
                frequencies[word] = 1
            else:
                frequencies[word] += 1
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
    if not check_object(candidate_keyword_phrases, list, tuple) \
            or not check_object(content_words, list):
        return None
    word_degrees = {}
    for word in content_words:
        word_degrees[word] = 0
    for phrase in candidate_keyword_phrases:
        for word in content_words:
            if word in phrase:
                word_degrees[word] += len(phrase)
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
    if not check_object(word_degrees, dict) \
            or not check_object(word_frequencies, dict):
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
    if not check_object(candidate_keyword_phrases, list, tuple) \
            or not check_object(word_scores, dict):
        return None
    cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score[phrase] = 0
        for word in phrase:
            if word not in word_scores.keys():
                return None
            cumulative_score[phrase] += word_scores[word]
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
    if not check_object(keyword_phrases_with_scores, dict) \
            or not check_positive_int(top_n) or not check_positive_int(max_length):
        return None
    for phrase in keyword_phrases_with_scores:
        if not check_object(phrase, tuple):
            return None
    top_with_long_phrases = sorted(
        keyword_phrases_with_scores.keys(), key=lambda key: keyword_phrases_with_scores[key],
        reverse=True)
    top_result = []
    for phrase in top_with_long_phrases:
        if len(phrase) <= max_length:
            top_result.append(' '.join(phrase))
    return top_result[:top_n]


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
    if not check_object(candidate_keyword_phrases, list, tuple) \
            or not check_object(phrases, list):
        return None
    keyword_pairs = list(pairwise(candidate_keyword_phrases))
    valid_keyword_pairs = [keyword_pair for keyword_pair in set(keyword_pairs)
                           if keyword_pairs.count(keyword_pair) >= 2]
    adjoined_phrases = []
    for phrase in phrases:
        phrase_list = phrase.lower().split()
        for keyword_pair in valid_keyword_pairs:
            pair_beg = keyword_pair[0]
            pair_end = keyword_pair[1]
            if ' '.join(pair_beg) not in phrase or ' '.join(pair_end) not in ' '.join(phrase_list):
                continue
            for index in range(len(phrase_list) - 2):
                if phrase_list[index] == pair_beg[-1] and phrase_list[index + 2] == pair_end[0]:
                    adjoined_phrase = phrase_list[index:index + 4]
                    if ' '.join(adjoined_phrase) in ' '.join(phrase_list):
                        adjoined_phrases.append(tuple(adjoined_phrase))
    valid_adjoined_phrases = [adjoined_phrase for adjoined_phrase in set(adjoined_phrases)
                              if adjoined_phrases.count(adjoined_phrase) >= 2]
    return valid_adjoined_phrases


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
    if not check_object(candidate_keyword_phrases, list, tuple) \
            or not check_object(word_scores, dict) \
            or not check_object(stop_words, list):
        return None
    cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score[phrase] = 0
        for word in phrase:
            if word not in stop_words:
                cumulative_score[phrase] += word_scores[word]
    return cumulative_score


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    if not text or not isinstance(text, str) or not check_positive_int(max_length):
        return None
    tokens = clean_and_tokenize(text)
    if not tokens:
        return None
    frequencies = calculate_frequencies(tokens)
    frequencies_sorted = sorted(frequencies.values())
    percentile_80 = frequencies_sorted[round(len(frequencies_sorted) / 100 * 80) - 1]
    stop_words = []
    for token, value in frequencies.items():
        if value >= percentile_80 and len(token) <= max_length:
            stop_words.append(token)
    return stop_words


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not path or not isinstance(path, Path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        stop_words = dict(json.load(f))
    return stop_words


def find_keyword_phrases(text: str, stop_words: Sequence[str]) -> None:
    """
    Finds keyword phrases without stop words in the given text and returns top n of these phrases.
    """
    candidate_keyword_phrases, frequencies, word_degrees, word_scores, \
        cumulative_score, candidate_with_stop_words, cumulative_with_stop_words = [None for i in range(7)]

    phrases = extract_phrases(text)

    if phrases:
        candidate_keyword_phrases = extract_candidate_keyword_phrases(phrases, stop_words)

    if candidate_keyword_phrases:
        frequencies = calculate_frequencies_for_content_words(candidate_keyword_phrases)

    if candidate_keyword_phrases and frequencies:
        word_degrees = calculate_word_degrees(candidate_keyword_phrases, list(frequencies.keys()))

    if frequencies and word_degrees:
        word_scores = calculate_word_scores(word_degrees, frequencies)

    if candidate_keyword_phrases and word_scores:
        cumulative_score = calculate_cumulative_score_for_candidates(candidate_keyword_phrases, word_scores)

    if cumulative_score:
        print('Top without stop words:', get_top_n(cumulative_score, 10, 5))

    if phrases and candidate_keyword_phrases:
        candidate_with_stop_words = extract_candidate_keyword_phrases_with_adjoining(
            candidate_keyword_phrases, phrases)

    if candidate_with_stop_words and word_scores and stop_words:
        cumulative_with_stop_words = calculate_cumulative_score_for_candidates_with_stop_words(
            candidate_with_stop_words, word_scores, stop_words)

    if cumulative_with_stop_words:
        print('Top with stop words:', get_top_n(cumulative_with_stop_words, 10, 5))
