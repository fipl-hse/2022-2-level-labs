"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
from re import split as rsplit, sub
from json import load

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def type_check(data: Any, expected: Any) -> bool:
    """
    Checks any type used in a program. For str, list, tuple and dict also checks if they are empty.
    Parameters:
    data (Any): An object which type is checked
    expected (Any): A type we expect data to be
    Returns:
    bool: True if data has the expected type and not empty, False otherwise
    """
    return isinstance(data, expected) and not (expected == int and isinstance(data, bool)) and data


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not type_check(text, str):
        return None
    punctuation = r"[–—!¡\"“”#$%&'()⟨⟩«»*+,./:;‹›<=>?¿@\]\[\\_`{|}~…⋯-]+"
    return [clean for phrase in rsplit(''.join((punctuation, r"(?=[$\s])|(?!=[\w\d])", punctuation)), text)
            if (clean := phrase.strip())]


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not type_check(phrases, list) or not type_check(stop_words, list):
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
    if not type_check(candidate_keyword_phrases, list):
        return None
    return {token: sum(look.count(token) for look in candidate_keyword_phrases)
            for phrase in set(candidate_keyword_phrases) for token in set(phrase)}


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
    if not type_check(candidate_keyword_phrases, list) or not type_check(content_words, list):
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
    if not type_check(word_degrees, dict) or not type_check(word_frequencies, dict) \
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
    if not type_check(candidate_keyword_phrases, list) or not type_check(word_scores, dict) or \
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
    if not type_check(keyword_phrases_with_scores, dict) \
            or not type_check(top_n, int) or top_n <= 0 or not type_check(max_length, int) or max_length <= 0:
        return None
    return sorted([' '.join(item) for item in keyword_phrases_with_scores if len(item) <= max_length],
                  key=lambda phrase: keyword_phrases_with_scores[tuple(phrase.split())], reverse=True)[:top_n]


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
    if not type_check(candidate_keyword_phrases, list) or not type_check(phrases, list):
        return None
    possible_pairs = {(sample, candidate_keyword_phrases[marker1+1])
                      for marker1, sample in enumerate(candidate_keyword_phrases[:-3])
                      for marker2, phrase in enumerate(candidate_keyword_phrases[marker1+2:-1])
                      if phrase == sample
                      and candidate_keyword_phrases[marker2 + 1] == candidate_keyword_phrases[marker1 + 1]}
    possible_phrases = [phrase[index:index+len1+len2+1]
                        for pair, len1, len2 in [(p_pair, len(p_pair[0]), len(p_pair[1])) for p_pair in possible_pairs]
                        for phrase in [tuple(phrase.lower().split()) for phrase in phrases]
                        for index in range(len(phrase[:-(len1 + len2)]))
                        if phrase[index:index+len1] == pair[0] and phrase[index+len1+1:index+len1+len2+1] == pair[1]]
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
    if not type_check(candidate_keyword_phrases, list) or \
            not type_check(word_scores, dict) or not type_check(stop_words, list):
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
    if not type_check(text, str) or not type_check(max_length, int) or max_length <= 0:
        return None
    punctuation = r"[–—!¡\"“”#$%&'()⟨⟩«»*+,./:;‹›<=>?¿@\]\[\\_`{|}~…⋯-]+"
    tokens = sub(''.join((punctuation, r"(?=[$\s])|(?!=[\w\d])", punctuation)), '', text).lower().split()
    frequencies = {token: tokens.count(token) for token in set(tokens)}
    percent_80 = sorted(frequencies.values(), reverse=True)[int(len(frequencies) * 0.2)]
    return [token for token in sorted(frequencies) if frequencies[token] >= percent_80 and len(token) <= max_length]


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not type_check(path, Path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return dict(load(file))


def process_text(text: str, stop_words: Optional[Sequence[str]] = None, max_length: Optional[int] = None) \
        -> Optional[Mapping[KeyPhrase, float]]:
    """
    Uses previous functions to process a text and extract key phrases.
    Accepts raw text and a list of stop words (or a maximum length of a stop word if they have to be generated
    from the text).
    Returns extracted key phrases or None if something goes wrong.
    """
    candidate_keyword_phrases, word_frequencies, word_degrees, word_scores, keyword_phrases_with_scores, \
        candidates_adjoined, cumulative_score_with_stop_words = \
        [None for not_undefined in range(7)]
    phrases = extract_phrases(text)
    if not stop_words and max_length and (stop_words_generated := generate_stop_words(text, max_length)):
        stop_words = stop_words_generated
    if phrases and stop_words:
        candidate_keyword_phrases = extract_candidate_keyword_phrases(phrases, stop_words)
    if candidate_keyword_phrases:
        word_frequencies = calculate_frequencies_for_content_words(candidate_keyword_phrases)
    if candidate_keyword_phrases and word_frequencies:
        word_degrees = calculate_word_degrees(candidate_keyword_phrases, list(word_frequencies.keys()))
    if word_degrees and word_frequencies:
        word_scores = calculate_word_scores(word_degrees, word_frequencies)
    if candidate_keyword_phrases and word_scores:
        keyword_phrases_with_scores = calculate_cumulative_score_for_candidates(candidate_keyword_phrases, word_scores)
    if candidate_keyword_phrases and phrases:
        candidates_adjoined = \
            extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases, phrases)
    if candidates_adjoined and word_scores and stop_words:
        cumulative_score_with_stop_words = \
            calculate_cumulative_score_for_candidates_with_stop_words(candidates_adjoined, word_scores, stop_words)
    else:
        cumulative_score_with_stop_words = {}
    if keyword_phrases_with_scores and cumulative_score_with_stop_words is not None:
        return {**keyword_phrases_with_scores, **cumulative_score_with_stop_words}
    return None
