"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
import re
from itertools import pairwise
import json
from lab_1_keywords_tfidf.main import check_positive_int

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def is_not_empty_or_bad(user_input: Any, element_type: Any, token_type: Any = None) -> bool:
    """
    checks whether the variable is empty or of the wrong type
    """
    if not user_input:
        return False
    if not isinstance(user_input, element_type):
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
    if not is_not_empty_or_bad(text, str):
        return None
    last = re.split(re.compile(re.compile(r"(?<=^)[^\s\w]+"r"|(?<=\s)[^\s\w]+"r"|[^\s\w]+(?=\s)"r"|[^\s\w]+(?=$)")
                               ), text.strip().replace("\n", ""))
    phrases = [i.strip() for i in last if i.strip()]
    return phrases


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not is_not_empty_or_bad(phrases, list) or not is_not_empty_or_bad(stop_words, list):
        return None
    phrase_list = []
    tuple_list = []
    candidate_keyword_phrases = []
    for phrase in phrases:
        phrase = phrase.lower()
        phrase_list.append(phrase.split())
    for phrase in phrase_list:  # (expression has type "List[str]", variable has type "str")  [assignment]
        for word in phrase:
            if word not in stop_words:
                tuple_list.append(word)
                continue
            phrase_tuple = tuple(tuple_list)
            if phrase_tuple:
                candidate_keyword_phrases.append(phrase_tuple)
                tuple_list.clear()
        if tuple_list:
            candidate_keyword_phrases.append(tuple(tuple_list))
            tuple_list.clear()
    return candidate_keyword_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not is_not_empty_or_bad(candidate_keyword_phrases, list, tuple):
        return None
    content_words_list = [item for every_tuple in candidate_keyword_phrases for item in every_tuple]
    content_words_freqs = {}
    for word in content_words_list:
        content_words_freqs[word] = content_words_list.count(word)
    return content_words_freqs


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
    if not is_not_empty_or_bad(candidate_keyword_phrases, list, tuple) \
            or not is_not_empty_or_bad(content_words, list):
        return None
    word_degrees = {}
    for word in content_words:
        tuples_with_word = [item for item in candidate_keyword_phrases if word in item]
        tuples_with_word_smushed = [item for every_tuple in tuples_with_word for item in every_tuple]
        word_deg_num = len(tuples_with_word_smushed)
        word_degrees[word] = word_deg_num
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
    if not is_not_empty_or_bad(word_degrees, dict) or not is_not_empty_or_bad(word_frequencies, dict)\
            or not all(word_frequencies.get(token, False) for token in word_degrees):
        return None
    key_for_word_score = list(word_degrees.keys())
    val_for_word_score = [a / b for a, b in zip(word_degrees.values(), word_frequencies.values())]
    word_scores = {key_for_word_score[i]: val_for_word_score[i] for i in range(len(key_for_word_score))}
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
    if not is_not_empty_or_bad(candidate_keyword_phrases, list, tuple) or not is_not_empty_or_bad(word_scores, dict):
        return None
    for tuple_phrase in candidate_keyword_phrases:
        for word in tuple_phrase:
            if word not in word_scores.keys():
                return None
    keyword_phrases_with_scores = dict.fromkeys(candidate_keyword_phrases)
    pure_score = 0.0
    for key in keyword_phrases_with_scores:
        for word in key:
            if word in word_scores:
                pure_score += word_scores.get(word)
                # Unsupported operand types for + ("int" and "None")  [operator]
        keyword_phrases_with_scores[key] = pure_score
        pure_score = 0.0
    return keyword_phrases_with_scores  # Incompatible return value type (got "Dict[Tuple[str, ...],
    # Optional[Any]]", expected "Optional[Mapping[Tuple[str, ...], float]]")  [return-value]


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
    if not is_not_empty_or_bad(keyword_phrases_with_scores, dict) \
            or not check_positive_int(top_n) or not check_positive_int(max_length):
        return None
    for phrase in keyword_phrases_with_scores:
        if not is_not_empty_or_bad(phrase, tuple):
            return None
    top_phrases = sorted([" ".join(phrase) for phrase in keyword_phrases_with_scores if len(phrase) <= max_length],
                         key=lambda phrase: keyword_phrases_with_scores[tuple(phrase.split())], reverse=True)[:top_n]
    return top_phrases


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
    if not is_not_empty_or_bad(candidate_keyword_phrases, list, tuple) or not is_not_empty_or_bad(phrases, list):
        return None
    keyword_pairs = list(pairwise(candidate_keyword_phrases))
    good_keyword_pairs = [keyword_pair for keyword_pair in set(keyword_pairs)
                          if keyword_pairs.count(keyword_pair) >= 2]
    adjoined_phrases = []
    for phrase in phrases:
        phrase = phrase.lower()
        phrase_list = phrase.split()
        for keyword_pair in good_keyword_pairs:
            for part_1, part_2 in pairwise(keyword_pair):
                part_1_str, part_2_str = ' '.join(part_1), ' '.join(part_2)
                if part_1_str not in phrase or part_2_str not in phrase:
                    continue
                for index in range(len(phrase_list) - 2):
                    if phrase_list[index] == part_1[-1] and phrase_list[index + 2] == part_2[0]:
                        adjoined_phrase = f"{part_1_str} {phrase_list[index + 1]} {part_2_str}".split()
                        if ' '.join(adjoined_phrase) in phrase:
                            adjoined_phrases.append(tuple(adjoined_phrase))
    good_adjoined_phrases = [adjoined_phrase for adjoined_phrase in set(adjoined_phrases)
                             if adjoined_phrases.count(adjoined_phrase) >= 2]
    return good_adjoined_phrases


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
    if not is_not_empty_or_bad(candidate_keyword_phrases, list, tuple) \
            or not is_not_empty_or_bad(word_scores, dict) \
            or not is_not_empty_or_bad(stop_words, list):
        return None
    all_phrases_scores = {}
    for phrase in candidate_keyword_phrases:
        all_phrases_scores[phrase] = sum(word_scores[word] for word in phrase if word not in stop_words)
    return all_phrases_scores


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    if not is_not_empty_or_bad(text, str) or not check_positive_int(max_length):
        return None
    text_no_commas = re.split(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]+\B', text.strip().replace("\n", ""))
    all_words = []
    for token in text_no_commas:
        word_list = token.strip().split(" ")
        all_words += word_list
    all_words_freqs = {}
    for word in all_words:
        all_words_freqs[word] = all_words.count(word)
    percent_80 = sorted(all_words_freqs.values(), reverse=True)[int(len(all_words_freqs) * 0.2)]
    stop_words = []
    for key, value in all_words_freqs.items():
        if value >= percent_80 and not len(key) > max_length:
            stop_words.append(key)
    return stop_words


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not is_not_empty_or_bad(path, Path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        array = json.load(f)
    return dict(array)  # Returning Any from function declared to return
    # "Optional[Mapping[str, Sequence[str]]]"  [no-any-return]


def process_text(text: str, stop_words: Optional[Sequence[str]] = None, max_length: Optional[int] = None) \
        -> Optional[Mapping[KeyPhrase, float]]:
    """
    Uses previous functions to process a text and extract key phrases.
    Accepts raw text and stop words list (or maximum length of a stop word if they have to be generated
    from the text).
    Returns extracted key phrases or None if something goes wrong.
    """
    candidate_keyword_phrases, content_words_freqs, word_degrees, word_scores, keyword_phrases_with_scores,\
        good_adjoined_phrases, all_phrases_scores = [None for _ in range(7)]

    phrases = extract_phrases(text)

    if not stop_words and not max_length:
        stop_words = generate_stop_words(text, 10)  # error: Argument 2 to "generate_stop_words"
        # has incompatible type "Optional[int]"; expected "int"  [arg-type]

    if phrases and stop_words:
        candidate_keyword_phrases = extract_candidate_keyword_phrases(phrases, stop_words)
    if candidate_keyword_phrases:
        content_words_freqs = calculate_frequencies_for_content_words(candidate_keyword_phrases)
    if candidate_keyword_phrases and content_words_freqs:
        word_degrees = calculate_word_degrees(candidate_keyword_phrases, list(content_words_freqs))
    if word_degrees and content_words_freqs:
        word_scores = calculate_word_scores(word_degrees, content_words_freqs)
    if candidate_keyword_phrases and word_scores:
        keyword_phrases_with_scores = calculate_cumulative_score_for_candidates(candidate_keyword_phrases, word_scores)
    if candidate_keyword_phrases and phrases:
        good_adjoined_phrases = (
            extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases, phrases))
    if good_adjoined_phrases and word_scores and stop_words:
        all_phrases_scores = (
            calculate_cumulative_score_for_candidates_with_stop_words(good_adjoined_phrases, word_scores, stop_words))
    else:
        all_phrases_scores = {}
    if keyword_phrases_with_scores and all_phrases_scores is not None:
        return {**keyword_phrases_with_scores, **all_phrases_scores}

    return None
