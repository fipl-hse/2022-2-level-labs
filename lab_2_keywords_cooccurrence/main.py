"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
import re

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def type_check(data: Any, expected: Any, not_empty: bool = False) -> bool:
    """
    Checks any type used in a program.
    Parameters:
    data (Any): An object which type is checked
    expected (Any): A type we expect data to be
    not_empty (bool): If expected is an str, a list or a dict, True stands for "it should not be empty" (optional)
    Returns:
    bool: True if data has the expected type and emptiness, False otherwise
    """
    return not (not isinstance(data, expected) or expected == int and isinstance(data, bool)) \
        and not (expected in (str, list, tuple, dict) and not_empty and not data)


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not type_check(text, str, True):
        return None
    punctuation = r"[–—!¡\"“”#$%&'()⟨⟩«»*+,./:;‹›<=>?¿@\]\[\\_`{|}~…⋯-]+"
    return [clean for phrase in re.split(''.join(
        (punctuation, r"(?=$|\s)|(?<=\s)", punctuation, r"|^", punctuation)), text) if (clean := phrase.strip())]


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not type_check(phrases, list, True) or not type_check(stop_words, list, True):
        return None
    candidates = []
    for phrase in phrases:
        clean_phrase = phrase.lower().split()
        splits = [-1] + [count for count, word in enumerate(clean_phrase) if word in stop_words] + [len(clean_phrase)]
        candidates.extend(candidate for count, split in enumerate(splits[:-1])
                          if (candidate := tuple(clean_phrase[split+1:splits[count+1]])))
    return candidates


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not type_check(candidate_keyword_phrases, list, True) \
            or not all(type_check(candidate, tuple, True) for candidate in candidate_keyword_phrases):
        return None
    return {token: sum(look.count(token) for look in candidate_keyword_phrases)
            for phrase in candidate_keyword_phrases for token in set(phrase)}


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
    if not type_check(candidate_keyword_phrases, list, True) or \
            not type_check(content_words, list, True):
        return None
    return {token: sum(len(phrase) for phrase in candidate_keyword_phrases if token in phrase)
            for token in content_words}


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not type_check(word_degrees, dict, True) or not type_check(word_frequencies, dict, True) \
            or not all(word_frequencies.get(token, False) for token in word_degrees):
        return None
    return {token: word_degrees[token] / word_frequencies[token] for token in word_degrees}


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
    if not type_check(candidate_keyword_phrases, list, True) or \
            not type_check(word_scores, dict, True) or \
            not all(token in word_scores for phrase in candidate_keyword_phrases for token in phrase):
        return None
    return {phrase: sum(word_scores[token] for token in phrase) for phrase in candidate_keyword_phrases}


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
    if not type_check(keyword_phrases_with_scores, dict, True) \
            or not type_check(top_n, int) or top_n <= 0 or not type_check(max_length, int) or max_length <= 0:
        return None
    return sorted(list(' '.join(item) for item in keyword_phrases_with_scores if len(item) <= max_length),
                  reverse=True, key=lambda phrase: keyword_phrases_with_scores[tuple(phrase.split())])[:top_n]


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
    if not type_check(candidate_keyword_phrases, list, True) or not type_check(phrases, list, True):
        return None
    possible_pairs = []
    for number1, sample in enumerate(candidate_keyword_phrases[:-3]):
        for number2, phrase in enumerate(candidate_keyword_phrases[number1+2:-1]):
            if phrase == sample and candidate_keyword_phrases[number2+1] == candidate_keyword_phrases[number1+1] \
                    and (pair := tuple((sample, candidate_keyword_phrases[number1+1]))) not in possible_pairs:
                possible_pairs.append(pair)
    possible_phrases = []
    for pair in possible_pairs:
        len1, len2 = len(pair[0]), len(pair[1])
        for item in phrases:
            phrase = tuple(item.lower().split())
            for index in range(len(phrase[:-(len1+len2)])):
                if phrase[index:index+len1] == pair[0] and phrase[index+len1+1:index+len1+len2+1] == pair[1]:
                    possible_phrases.append(phrase[index:index+len1+len2+1])
    return [phrase for phrase in set(possible_phrases) if possible_phrases.count(phrase) > 1]


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
    if not type_check(candidate_keyword_phrases, list, True) or \
            not type_check(word_scores, dict, True) or not type_check(stop_words, list, True):
        return None
    return {phrase: sum(word_scores[token] for token in phrase if token not in stop_words)
            for phrase in candidate_keyword_phrases}


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
