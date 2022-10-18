"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping
import re
import json

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

    punctuation = r',;:¡!¿?…⋯‹›«»\\"“”\[\]()⟨⟩}{&]|[-–~—]'
    for punct_mark in punctuation:
        text = text.replace(punct_mark, '.')
    split_text = text.split('.')
    phrases = []
    for phrase in split_text:
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

    if not isinstance(phrases, list) or not phrases:
        return None
    if not isinstance(stop_words, list) or not stop_words:
        return None

    candidate_phrases = []
    for phrase in phrases:
        split_phrase = phrase.lower().split()
        no_stop_words = []
        for token in split_phrase:
            if token not in stop_words:
                no_stop_words.append(token)
            else:
                candidate_phrases.append(no_stop_words)
                no_stop_words = []
        candidate_phrases.append(no_stop_words)
        candidate_keyword_phrases = []
        for candidate_phrase in candidate_phrases:
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

    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None

    content_words = {}
    tokens = []
    for phrase in candidate_keyword_phrases:
        for token in phrase:
            tokens.append(token)
    for token in tokens:
        content_words[token] = tokens.count(token)

    return content_words



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

    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    if not isinstance(content_words, list) or not content_words:
        return None

    word_degrees = {}

    for phrase in candidate_keyword_phrases:
        for token in phrase:
            if token in content_words and token in word_degrees:
                word_degrees[token] += len(phrase)
            elif token in content_words:
                word_degrees[token] = len(phrase)
    for token in content_words:
        if token not in word_degrees.keys():
            word_degrees[token] = 0

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

    if not isinstance(word_degrees, dict) or not word_degrees:
        return None
    if not isinstance(word_frequencies, dict) or not word_frequencies:
        return None
    if not word_degrees.keys() == word_frequencies.keys():
        return None

    word_scores = {}

    for token in word_degrees.keys():
        word_scores[token] = word_degrees[token]/word_frequencies[token]

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

    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    if not isinstance(word_scores, dict) or not word_scores:
        return None

    keyword_phrases_with_scores = {}

    for phrase in candidate_keyword_phrases:
        keyword_phrases_with_scores[phrase] = 0
        for token in phrase:
            if token in word_scores.keys():
                keyword_phrases_with_scores[phrase] += word_scores[token]
            else:
                return None

    return keyword_phrases_with_scores


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

    if not isinstance(keyword_phrases_with_scores, dict) or not keyword_phrases_with_scores:
        return None
    if not isinstance(top_n, int) or not top_n > 0:
        return None
    if not isinstance(max_length, int) or not max_length > 0:
        return None

    sorted_phrases = sorted(keyword_phrases_with_scores.keys(), key=lambda x: keyword_phrases_with_scores[x],
                            reverse=True)

    max_length_list = []
    for phrase in sorted_phrases:
        phrase_length = len(phrase)
        if phrase_length <= max_length:
            joined_phrase = ' '.join(phrase)
            max_length_list.append(joined_phrase)

    return max_length_list[:top_n]


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

    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases or \
            not isinstance(phrases, list) or not phrases:
        return None

    pairs_list = []
    joined_phrases = [' '.join(phrase) for phrase in candidate_keyword_phrases]
    for index1, phrase1 in enumerate(joined_phrases):
        for index2, phrase2 in enumerate(joined_phrases):
            if index2 == index1 + 1:
                pair = phrase1, phrase2
                pairs_list.append(pair)
    pair_occurence = {}
    for pair in pairs_list:
        pair_occurence[pair] = pair_occurence.get(pair, 0) + 1
    frequent_pairs = {pair: occur for pair, occur in pair_occurence.items() if pair_occurence[pair] > 1}
    possible_candidate_phrases = []
    for pair in frequent_pairs:
        for phrase in phrases:
            if pair[0] in phrase and pair[1] in phrase:
                possible_candidate_phrases.extend(re.findall(f'{pair[0]} .* {pair[1]}', phrase))
    new_candidate_keyword_phrases = []
    for phrase in possible_candidate_phrases:
        if possible_candidate_phrases.count(phrase) > 1 and \
                tuple(phrase.split()) not in new_candidate_keyword_phrases:
            new_candidate_keyword_phrases.append(tuple(phrase.split()))
    return new_candidate_keyword_phrases


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

    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    if not isinstance(word_scores, dict) or not word_scores:
        return None
    if not isinstance(stop_words, list) or not stop_words:
        return None

    cumulative_scores = {}

    for phrase in candidate_keyword_phrases:
        score = 0.0
        for word in phrase:
            if word not in stop_words and word in word_scores.keys():
                score += word_scores[word]
        cumulative_scores[phrase] = score
    return cumulative_scores


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """

    if not isinstance(text, str) or not text:
        return None
    if not isinstance(max_length, int) or not max_length > 0:
        return None

    punctuation = r',;:¡!¿?…⋯‹›«»\\"“”\[\]()⟨⟩}{&]|[-–~—]'
    for punct_mark in punctuation:
        text = text.replace(punct_mark, '')
    word_list = text.lower().split()
    frequencies_dict = {}
    for word in word_list:
        frequencies_dict[word] = word_list.count(word)
    sorted_frequencies = sorted(frequencies_dict.values())
    stop_words_list = []
    percentile = int((80/100) * len(sorted_frequencies))
    for word, freq in frequencies_dict.items():
        if sorted_frequencies[percentile - 1] <= freq <= max_length:
            stop_words_list.append(word)
    return stop_words_list


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """

    if not isinstance(path, Path):
        return None

    with open(path, encoding='utf-8') as file:
        stop_words = json.load(file)

    return stop_words
