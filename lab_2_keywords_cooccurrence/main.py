"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any, Type
import re
from itertools import repeat, pairwise, chain
from json import load


KeyPhrase = tuple[str]
KeyPhrases = Sequence[KeyPhrase]


def type_check(data: Any, expected: Type) -> bool:
    """
    Checks any type used in the program.
    """
    return isinstance(data, expected) and not (expected == int and isinstance(data, bool)) and data


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters.
    """
    if not isinstance(text, str):
        return None
    expression = re.compile(r'(?<=^)[^\s\w]')
    return [clean for phrase in re.split(expression, text) if (clean := phrase.strip())]


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of phrases from the stop words.
    """
    if not isinstance(phrases, list) or not isinstance(stop_words, list):
        return None
    candidates = []
    for phrase in [phrase.lower().split() for phrase in phrases]:
        splits = [-1] + [index for index, word in enumerate(phrase) if word in stop_words] + [len(phrase)]
        candidates.extend(tuple(candidate) for start, end in pairwise(splits) if (candidate := phrase[start+1:end]))
    return candidates


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the words from the keyword phrases list and computes their frequencies.
    """
    if not isinstance(candidate_keyword_phrases, list):
        return None
    candidates_chained = list(chain.from_iterable(candidate_keyword_phrases))
    return {token: candidates_chained.count(token) for token in set(candidates_chained)}


def calculate_word_degrees(candidate_keyword_phrases: KeyPhrases,
                           content_words: Sequence[str]) -> Optional[Mapping[str, int]]:
    """
    Calculates the word degree based on keyword phrases.
    """
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(content_words, list):
        return None
    word_degrees = {}
    for token in content_words:
        word_degrees[token] = sum(len(phrase) for phrase in candidate_keyword_phrases if token in phrase)
    return word_degrees


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculates the word score based on the word degree and the word frequency.
    """

    if not isinstance(word_degrees, dict) or not isinstance(word_frequencies, dict) \
            or not all(word_frequencies.get(token, False) for token in word_degrees):
        return None
    return {token: word_degrees[token] / word_frequencies[token] for token in word_degrees}


def calculate_cumulative_score_for_candidates(candidate_keyword_phrases: KeyPhrases,
                                              word_scores: Mapping[str, float]) -> Optional[Mapping[KeyPhrase, float]]:
    """
    Calculates cumulative score for each keyword phrase.
    """
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(word_scores, dict) or \
            not all(token in word_scores for token in list(chain.from_iterable(candidate_keyword_phrases))):
        return None
    return {phrase: sum(word_scores[token] for token in phrase) for phrase in candidate_keyword_phrases}


def get_top_n(keyword_phrases_with_scores: Mapping[KeyPhrase, float],
              top_n: int,
              max_length: int) -> Optional[Sequence[str]]:
    """
    Extracts the top of keyword phrases based on their scores and lengths.
    """
    if not isinstance(keyword_phrases_with_scores, dict) \
            or not isinstance(top_n, int) or top_n <= 0 or not isinstance(max_length, int) or max_length <= 0:
        return None
    filtered = [item for item in keyword_phrases_with_scores if len(item) <= max_length]
    filtered_and_sorted = sorted(filtered, key=lambda phrase: keyword_phrases_with_scores[phrase], reverse=True)
    return [' '.join(item) for item in filtered_and_sorted][:top_n]


def extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases: KeyPhrases,
                                                     phrases: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Extracts the keyword phrases from the candidate keywords sequence and
    builds new containing stop words.
    """
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(phrases, list):
        return None
    pairs = list(pairwise(candidate_keyword_phrases))
    possible_pairs = [pair for pair in set(pairs) if pairs.count(pair) > 1]
    possible_phrases = []
    for pair, len1, len2 in [(pair, len(pair[0]), len(pair[1])) for pair in possible_pairs]:
        for phrase in [tuple(phrase.lower().split()) for phrase in phrases]:
            for start, stop_word, end in [(i, i+len1, i+len1+len2) for i in range(len(phrase)-len1-len2)]:
                if pair == (phrase[start:stop_word], phrase[stop_word+1:end+1]):
                    possible_phrases.append(phrase[start:end+1])
    return [phrase for phrase in set(possible_phrases) if possible_phrases.count(phrase) > 1]


def calculate_cumulative_score_for_candidates_with_stop_words(candidate_keyword_phrases: KeyPhrases,
                                                              word_scores: Mapping[str, float],
                                                              stop_words: Sequence[str]) \
        -> Optional[Mapping[KeyPhrase, float]]:
    """
    Calculates score for the keyword phrase.
    """
    if not isinstance(candidate_keyword_phrases, list) or \
            not isinstance(word_scores, dict) or not isinstance(stop_words, list):
        return None
    cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score[phrase] = sum(word_scores[token] for token in phrase if token not in stop_words)
    return cumulative_score


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Chooses the list of stop words from the text.
    """
    if not isinstance(text, str) or not isinstance(max_length, int) or max_length <= 0:
        return None
    expression = re.compile(r"(?<=^)[^\s\w]+")
    tokens = re.sub(expression, '', text).lower().split()
    frequencies = {token: tokens.count(token) for token in set(tokens)}
    percent_80 = sorted(frequencies.values(), reverse=True)[int(len(frequencies) * 0.2)]
    return [token for token in sorted(frequencies) if frequencies[token] >= percent_80 and len(token) <= max_length]


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file.
    """
    if not isinstance(path, Path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return dict(load(file))
    with open(path, 'r', encoding='utf-8') as file:
        return dict(load(file))


def process_text(text: str, stop_words: Optional[Sequence[str]] = None, max_length: Optional[int] = None) -> Optional[Mapping[KeyPhrase, float]]:
    """
    Extract key phrases, returns extracted key phrases or None.
    """
    candidate_keyword_phrases, word_frequencies, word_degrees, word_scores, keyword_phrases_with_scores, \
        candidates_adjoined, cumulative_score_with_stop_words = repeat(None, 7)
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
