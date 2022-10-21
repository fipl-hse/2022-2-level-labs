"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping
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

    punctuation = r'''.,;:¡!¿?…⋯‹›«»\\"“”\[\]()⟨⟩}{&]|[-–~—]'''
    for i in text:
        if i in punctuation:
            text = text.replace(i, ',')
    list_of_tokens = text.split(',')
    phrases = []
    for token in list_of_tokens:
        phrase = token.strip()
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

    candidate_keyword_phrases = []

    for phrase in phrases:
        tokens = phrase.lower()
        tokens = tokens.split()
        list_of_tokens = []
        for token in tokens:
            if token not in stop_words:
                list_of_tokens.append(token)
            elif list_of_tokens:
                candidate_keyword_phrases.append(tuple(list_of_tokens))
                list_of_tokens = []
        if list_of_tokens:
            candidate_keyword_phrases.append(tuple(list_of_tokens))

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

    frequencies_for_content_words = {}

    for phrase in candidate_keyword_phrases:
        for token in phrase:
            frequencies_for_content_words[token] = frequencies_for_content_words.get(token, 0) + 1

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
    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    if not isinstance(content_words, list) or not content_words:
        return None

    word_degrees = {}

    for phrase in candidate_keyword_phrases:
        for word in content_words:
            if word not in word_degrees.keys():
                word_degrees[word] = 0
            if word in phrase:
                word_degrees[word] = len(phrase) + word_degrees[word]

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
    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    if not isinstance(word_scores, dict) or not word_scores:
        return None

    cumulative_score_for_candidates = {}

    for phrase in candidate_keyword_phrases:
        phrase_score = 0.0
        for word in phrase:
            if word not in word_scores:
                return None
            phrase_score += word_scores[word]
        cumulative_score_for_candidates[phrase] = phrase_score

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
    if not isinstance(keyword_phrases_with_scores, dict) or not keyword_phrases_with_scores:
        return None
    if not isinstance(top_n, int) or not top_n > 0:
        return None
    if not isinstance(max_length, int) or not max_length > 0:
        return None

    top = sorted(keyword_phrases_with_scores.keys(), key=lambda key: keyword_phrases_with_scores[key], reverse=True)
    top_phrases = []
    for phrase in top:
        if len(phrase) <= max_length:
            top_phrases.append(' '.join(phrase))

    return top_phrases[:top_n]


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
    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases or\
            not isinstance(phrases, list) or not phrases:
        return None

    new_keyword_phrases = []
    all_pairs = []
    for i in range(len(candidate_keyword_phrases) - 1):
        pair = []
        pair.append(candidate_keyword_phrases[i])
        pair.append(candidate_keyword_phrases[i+1])
        all_pairs.append(pair)
    important_pairs = []
    for pair in all_pairs:
        if all_pairs.count(pair) > 1 and pair not in important_pairs:
            important_pairs.append(pair)
    for part_1, part_2 in important_pairs:
        part1 = ' '.join(part_1)
        part2 = ' '.join(part_2)
        for phrase in phrases:
            phrase = phrase.lower()
            if part1 not in phrase or part2 not in phrase:
                continue
            tokens = phrase.split()
            last_word_in_part1 = part1.split()[-1] if ' ' in part1 else part1
            first_word_in_part2 = part2.split()[0] if ' ' in part2 else part2
            for i in range(len(tokens) - 2):
                if tokens[i] != last_word_in_part1 or tokens[i+2] != first_word_in_part2:
                    continue
                new_phrase = part1 + ' ' + tokens[i+1] + ' ' + part2
                if new_phrase in phrase:
                    new_keyword_phrases.append(tuple(new_phrase.split()))
    keyword_phrases = []
    for key in new_keyword_phrases:
        if key in keyword_phrases or new_keyword_phrases.count(key) < 2:
            continue
        keyword_phrases.append(key)

    return keyword_phrases


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
        phrase_score = 0.0
        for word in phrase:
            if word in stop_words:
                continue
            if word not in word_scores.keys():
                return None
            phrase_score += word_scores[word]
        cumulative_scores[phrase] = phrase_score

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

    perzentil = 0.8
    punctuation = r'''.,;:¡!¿?…⋯‹›«»\\"“”\[\]()⟨⟩}{&]|[-–~—]'''
    for i in text:
        if i in punctuation:
            text = text.replace(i, '').lower()
    tokens = text.split()
    frequencies = {token: tokens.count(token) for token in tokens}
    sorted_freqs = sorted(frequencies.values())
    rank = sorted_freqs[int(perzentil * len(sorted_freqs) - 1)]
    list_of_stop_words = []
    for word, freq in frequencies.items():
        if freq >= rank and len(word) <= max_length:
            list_of_stop_words.append(word)

    return list_of_stop_words


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not isinstance(path, Path):
        return None

    # with open(path, encoding='utf-8') as file_with_stopwords:
    with path.open(encoding='utf-8') as file_with_stopwords:
        dict_of_stop_words = json.load(file_with_stopwords)
    if not isinstance(dict_of_stop_words, dict):
        return None

    return dict_of_stop_words
