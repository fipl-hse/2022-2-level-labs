"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
import json
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
import re
from itertools import chain, pairwise
from lab_1_keywords_tfidf.main import clean_and_tokenize, calculate_frequencies

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_type_and_not_empty(some_object: Any, some_type: type) -> bool:
    """
    Checks object type and whether it contains something
    """
    return bool(some_object and isinstance(some_object, some_type))


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not check_type_and_not_empty(text, str):
        return None
    pattern = re.compile(r'''
                        [^\w\s]+(?!\w)   # first pattern matches punctuation that is not followed by alphanumeric characters
                        |
                         (?<!\w)[^\w\s]+  # second pattern matches punctuation that is not preceeded by alphanumeric characters
                         ''', re.VERBOSE)
    return [clean_phrase for phrase in re.split(pattern, text) if (clean_phrase := phrase.strip())]


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not check_type_and_not_empty(phrases, list) or not check_type_and_not_empty(stop_words, list):
        return None
    text = ' ? '.join(phrases)
    tokens = text.lower().split()
    without_stopwords = [token if token not in stop_words else '?' for token in tokens]
    new_text = ' '.join(without_stopwords)
    return [tuple(phrase.split()) for phrase in extract_phrases(new_text)]


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_type_and_not_empty(candidate_keyword_phrases, list):
        return None
    words = list(chain(*candidate_keyword_phrases))
    return calculate_frequencies(words)


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
    if (not check_type_and_not_empty(candidate_keyword_phrases, list)
            or not check_type_and_not_empty(content_words, list)):
        return None
    word_degree = dict.fromkeys(content_words, 0)
    for word in content_words:
        for phrase in candidate_keyword_phrases:
            if word not in phrase:
                continue
            word_degree[word] += len(phrase)
    return word_degree


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if (not check_type_and_not_empty(word_degrees, dict)
            or not check_type_and_not_empty(word_frequencies, dict)
            or word_degrees.keys() != word_frequencies.keys()):
        return None
    word_score = {word: word_degrees[word] / word_frequencies[word] for word in word_degrees}
    return word_score


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
    if (not check_type_and_not_empty(candidate_keyword_phrases, list)
            or not check_type_and_not_empty(word_scores, dict)):
        return None
    cumulative_scores = {}
    for phrase in candidate_keyword_phrases:
        try:
            score = sum(word_scores[word] for word in phrase)
        except KeyError:
            return None
        cumulative_scores[phrase] = score
    return cumulative_scores


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
    if (not check_type_and_not_empty(top_n, int)
            or not check_type_and_not_empty(max_length, int)
            or max_length < 0
            or not check_type_and_not_empty(keyword_phrases_with_scores, dict)
            or top_n < 0):
        return None
    normal_length_phrases = [phrase for phrase in keyword_phrases_with_scores if len(phrase) <= max_length]
    sorted_phrases = sorted(normal_length_phrases,
                            key=lambda x: keyword_phrases_with_scores[x],
                            reverse=True)
    return [' '.join(words) for words in sorted_phrases[:top_n]]


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
    if not check_type_and_not_empty(candidate_keyword_phrases, list) or not check_type_and_not_empty(phrases, list):
        return None

    pairs = list(pairwise(candidate_keyword_phrases))
    neighbours = [pair for pair in set(pairs) if pairs.count(pair) > 1]

    phrases = [phrase.lower() for phrase in phrases]
    keyword_phrases_with_stop_words = []
    for phrase in phrases:
        for pair in neighbours:
            part1, part2 = ' '.join(pair[0]), ' '.join(pair[1])
            keyword_phrases_with_stop_words.extend(re.findall(fr'\b{part1}\s\w+\s{part2}\b', phrase))

    return [tuple(phrase.split()) for phrase in set(keyword_phrases_with_stop_words)
            if keyword_phrases_with_stop_words.count(phrase) > 1]


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
    if (not check_type_and_not_empty(candidate_keyword_phrases, list)
            or not check_type_and_not_empty(word_scores, dict)
            or not check_type_and_not_empty(stop_words, list)):
        return None
    cumulative = {}
    for phrase in candidate_keyword_phrases:
        cumulative[phrase] = sum(word_scores[word] for word in phrase if word not in stop_words)
    return cumulative


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    if not check_type_and_not_empty(text, str) or not check_type_and_not_empty(max_length, int) or max_length < 0:
        return None
    tokens = clean_and_tokenize(text)
    if tokens:
        frequencies = calculate_frequencies(tokens)
    else:
        return None
    scores = sorted(frequencies.values())
    percent = 80
    percentile = scores[round(len(scores) * percent / 100) - 1]
    return [phrase for phrase, score in frequencies.items()
            if score >= percentile and len(phrase) <= max_length]


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not check_type_and_not_empty(path, Path):
        return None
    with path.open(encoding='utf-8') as file_with_stopwords:
        stop_words = json.load(file_with_stopwords)
    if isinstance(stop_words, dict):
        return stop_words
    return None
