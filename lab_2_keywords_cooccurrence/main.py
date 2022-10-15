"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
from itertools import pairwise
import re
from lab_1_keywords_tfidf.main import check_positive_int


KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_types(user_var: Any, expected_type: Any) -> bool:
    """
    Checks type of variable and compares it with expected type
    """
    if not (isinstance(user_var, expected_type) and user_var):
        return False
    if expected_type == list or Sequence:
        for element in user_var:
            if not element:
                return False
    elif expected_type == dict or Mapping:
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
    if not (check_types(text, str) and text):
        return None
    separators = ',;:¡!¿?…⋯‹›«»\\"“”[]()⟨⟩}{&|-–~—'
    for separator in separators:
        text = text.replace(separator, '.')
    split_text = text.split('.')
    new_split_text = []
    for phrase in split_text:
        phrase = phrase.strip()
        if phrase:
            new_split_text.append(phrase)
    return new_split_text


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not (check_types(phrases, Sequence) and check_types(stop_words, Sequence)):
        return None
    tuples_candidate_phrases = []
    candidate_phrases = []
    for phrase in phrases:
        split_phrase = phrase.lower().split()
        temp_candidate_phrase = []
        for word in split_phrase:
            if word not in stop_words:
                temp_candidate_phrase.append(word)
            else:
                candidate_phrases.append(temp_candidate_phrase)
                temp_candidate_phrase = []
        candidate_phrases.append(temp_candidate_phrase)
        tuples_candidate_phrases = []
        for candidate_phrase in candidate_phrases:
            if candidate_phrase:
                tuples_candidate_phrases.append(tuple(candidate_phrase))
    return tuples_candidate_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_types(candidate_keyword_phrases, Sequence):
        return None
    tokens = []
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            tokens.append(word)
    freq_dict = {token: tokens.count(token) for token in set(tokens)}
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
    if not (check_types(candidate_keyword_phrases, Sequence) and check_types(content_words, Sequence)):
        return None
    word_degrees_dict = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word in word_degrees_dict and word in content_words:
                word_degrees_dict[word] += len(phrase)
            elif word in content_words:
                word_degrees_dict[word] = len(phrase)
    for content_word in content_words:
        if content_word not in word_degrees_dict.keys():
            word_degrees_dict[content_word] = 0
    return word_degrees_dict


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not (check_types(word_degrees, Mapping) and check_types(word_frequencies, Mapping)
            and word_degrees.keys() == word_frequencies.keys()):
        return None
    return {word: word_degrees[word] / word_frequencies[word] for word in word_degrees.keys()}


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
    if not (check_types(candidate_keyword_phrases, Sequence) and check_types(word_scores, Mapping)):
        return None
    cumulative_score_dict = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score_dict[phrase] = 0
        for word in phrase:
            if word not in word_scores:
                return None
            cumulative_score_dict[phrase] += int(word_scores[word])
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
    if not (check_types(keyword_phrases_with_scores, Mapping)
            and check_positive_int(top_n)
            and check_positive_int(max_length)):
        return None
    top = sorted(keyword_phrases_with_scores.keys(),
                 key=lambda phrase: keyword_phrases_with_scores[phrase],
                 reverse=True)
    top_str = []
    for phrase in top:
        phrase_length = len(phrase)
        if  phrase_length <= max_length:
            phrase_join = ' '.join(phrase)
            top_str.append(phrase_join)
    return top_str[:top_n]


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
    if not (check_types(candidate_keyword_phrases, Sequence) and check_types(phrases, Sequence)):
        return None
    kw_phrases_join = [' '.join(kw_phrase) for kw_phrase in candidate_keyword_phrases]
    kw_phrases_pairs = list(pairwise(kw_phrases_join))
    frequencies = {token: kw_phrases_pairs.count(token) for token in kw_phrases_pairs}
    kw_phrases_with_stop_word = []
    for kw_phrase in frequencies.keys():
        if frequencies[kw_phrase] > 1:
            for phrase in phrases:
                if kw_phrase[0] in phrase and kw_phrase[1] in phrase:
                    kw_phrases_with_stop_word.extend(re.findall(f'{kw_phrase[0]}.*{kw_phrase[1]}', phrase))
    true_kw_phrases_with_stop_word = []
    for kw_phrase_with_stop_word in dict.fromkeys(kw_phrases_with_stop_word):
        if kw_phrases_with_stop_word.count(kw_phrase_with_stop_word) > 1:
            true_kw_phrases_with_stop_word.append(tuple(kw_phrase_with_stop_word.split()))
    return true_kw_phrases_with_stop_word


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
    if not (check_types(candidate_keyword_phrases, Sequence)
            and check_types(word_scores, Mapping)
            and check_types(stop_words, Sequence)):
        return None
    cumulative_score_with_stop_words_dict = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score_with_stop_words_dict[phrase] = 0
        for word in phrase:
            if word not in stop_words:
                cumulative_score_with_stop_words_dict[phrase] += int(word_scores[word])
    return cumulative_score_with_stop_words_dict


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
