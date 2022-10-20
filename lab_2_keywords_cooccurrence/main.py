"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
from string import punctuation
import json

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def correct_type(variable: Any, expected_type: type) -> bool:
    """
    Checks the type of variable
    """
    if not isinstance(variable, expected_type) or not variable:
        return False
    if isinstance(variable, int):
        if variable < 0:
            return False
    return True


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not correct_type(text, str):
        return None
    new_punctuation = punctuation + "¡¿…⋯‹›«»“”⟨⟩–—"
    for punctuation_mark in new_punctuation:
        text = text.replace(punctuation_mark, ",")
    return [word for word in [word.strip() for word in text.split(",")] if word]


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not correct_type(phrases, list) or not correct_type(stop_words, list):
        return None
    candidate_phrases = []
    new_list = [words.lower().split() for words in phrases]
    for token in new_list:
        indexes = [-1] + [index for index, item in enumerate(token) if item in stop_words]
        new_ind = 1
        for index in indexes[:-1]:
            if phrase := tuple(token[index + 1:indexes[new_ind]]):
                candidate_phrases.append(phrase)
            new_ind += 1
        if phrase := tuple(token[indexes[-1] + 1:]):
            candidate_phrases.append(phrase)
    return candidate_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not correct_type(candidate_keyword_phrases, list):
        return None
    freq_dict = {}
    for phrase in candidate_keyword_phrases:
        for token in phrase:
            if token in freq_dict:
                freq_dict[token] += 1
            else:
                freq_dict[token] = phrase.count(token)
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
    if not correct_type(candidate_keyword_phrases, list) or not correct_type(content_words, list):
        return None
    word_degree = {}
    for word in content_words:
        for i in candidate_keyword_phrases:
            if word in i:
                word_degree |= {word: word_degree.get(word, 0) + len(i)}
        if word not in word_degree:
            word_degree[word] = 0
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
    if not correct_type(word_degrees, dict) or not correct_type(word_frequencies, dict) \
            or not all(word_frequencies.get(word) for word in word_degrees.keys()):
        return None
    return {word: word_degrees[word] / word_frequencies[word] for word in word_degrees}


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
    if not correct_type(candidate_keyword_phrases, list) or not correct_type(word_scores, dict) \
            or not all(word_scores.get(word) for phrase in candidate_keyword_phrases for word in phrase):
        return None
    return {phrase: int(sum(word_scores[word] for word in phrase)) for phrase in candidate_keyword_phrases}


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
    if not correct_type(keyword_phrases_with_scores, dict) \
            or not correct_type(top_n, int) or not correct_type(max_length, int):
        return None
    keys = [key for key in keyword_phrases_with_scores if len(key) <= max_length]
    sorted_keys = sorted(keys, reverse=True, key=lambda phrase: keyword_phrases_with_scores[phrase])[:top_n]
    return [" ".join(item) for item in sorted_keys]


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
    if not correct_type(candidate_keyword_phrases, list) or not correct_type(phrases, list):
        return None
    join_phrases = [" ".join(phrase) for phrase in candidate_keyword_phrases]
    new_phrases = [tuple((join_phrases[ind: ind + 2])) for ind in range(len(join_phrases))]
    freq_dict = {token: new_phrases.count(token) for token in new_phrases}
    new_candidates = []
    for key in freq_dict:
        if freq_dict[key] < 2:
            continue
        for token in [" ".join(list(key)).split()]:
            for phrase in phrases:
                for ind in range(len(phrase)):
                    with_stop = phrase.lower().split()[ind:ind + len(key[0].split()) + len(key[1].split()) + 1]
                    if set(token).issubset(with_stop):
                        new_candidates.append(tuple(with_stop))
    for element in set(new_candidates):
        new_candidates.remove(element)
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
    if not correct_type(candidate_keyword_phrases, list) \
            or not correct_type(word_scores, dict) or not correct_type(stop_words, list):
        return None
    return {phrase: int(sum(word_scores[word] for word in phrase if word not in stop_words)) for phrase in
            candidate_keyword_phrases}


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    if not correct_type(text, str) or not correct_type(max_length, int):
        return None
    new_punctuation = punctuation + "¡¿…⋯‹›«»“”⟨⟩–—"
    for punctuation_mark in new_punctuation:
        text = text.replace(punctuation_mark, " ")
    tokens = text.lower().split()
    freq_dict = {token: tokens.count(token) for token in tokens}
    sorted_values = sorted(freq_dict.values())
    position_percentile_80 = int((80 / 100) * len(sorted_values))
    return [key for key, value in freq_dict.items() if sorted_values[position_percentile_80 - 1] <= value <= max_length]


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not isinstance(path, Path):
        return None
    with path.open(encoding='utf-8') as text:
        return dict(json.load(text))
