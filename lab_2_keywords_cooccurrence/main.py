"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
import re
import itertools
import json
from pathlib import Path
from typing import Optional, Sequence, Mapping, Union, Any, Type


KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def for_i_empty_checker(collection: Union[set, dict, list, tuple]) -> bool:
    """
    Check if collection's items are False

    Parameters:
    collection: Union[set, dict, list, tuple]: collection, which includes some items

    Returns:
    bool: True, if items are not False (not empty, or not 0, if numbers) and False otherwise
    """
    return bool(collection and all(collection))


def my_isinstance(instance: Any, type_of_instance: Type[Any]) -> bool:
    """
    Distincts int and bool compared to built-in isinstance() function.

    Parameters:
    instance: Any
    type_of_instance: Any

    Returns:
    bool: True if instance's type is expected type, False, if instance's type is bool and expected type
    """

    if type_of_instance is not int:
        return bool(isinstance(instance, type_of_instance) and instance)
    return bool(not isinstance(instance, bool) and isinstance(instance, int) and instance)


def for_i_type_checker(collection: Union[set, list, tuple],
                       type_of_collection: Type[Any],
                       type_of_instance: Type[Any]) -> bool:
    """
    Acts like my_isinstance for every collection's item

    Parameters:
    collection: Union[set, dict, list, tuple]
    type_of_collection: Union[set, dict, list, tuple]
    type_of_instance: Any

    Returns:
    bool: True if instance's type is expected type, False, if instance's
    type is bool and expected type is int or if expected collection's type
    doesn't equal collection's type
    """

    return (my_isinstance(collection, type_of_collection)
            and all(map(lambda x: my_isinstance(x, type_of_instance), collection)))


def is_dic_correct(dic: dict,
                   allow_false_items: bool,
                   key_type: Type[Any],
                   value_type: Any) -> bool:
    """
    Checks dictionary on being empty, having False items in keys and values,
    correspondence of keys and values to the types we expect to observe

    Parameters:
    dic: dict
    allow_false_items: bool
    key_type: Union[int, float, str, tuple]
    value_type: Any

    Returns:
    bool: True if dict is not empty,
    it's keys and values correspond to expected type
    and deprived of False items, else: False
    """

    if not my_isinstance(dic, dict):
        return False

    keys = list(dic.keys())
    values = list(dic.values())
    is_empty = bool(allow_false_items or dic)
    if is_empty:
        return (for_i_type_checker(keys, list, key_type)
                and for_i_type_checker(values, list, value_type))

    return (for_i_type_checker(keys, list, key_type) and for_i_type_checker(values, list, value_type)
            and for_i_empty_checker(keys) and for_i_empty_checker(values) and is_empty)


# CAN I GET AN EXTRA TASK TO HIGHER MY GRADE
# AND COMPENSATE THE FACT OFF MISSING DEADLINE?
def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """

    if not my_isinstance(text, str):
        return None
    pattern = re.compile(r"(?<=^)[^\s\w]+|(?<=\s)[^\s\w]+|[^\s\w]+(?=\s)|[^\s\w]+(?=$)")

    #  ?= -- positive lookahead     ?<= -- positive lookbehind
    #  ?! -- negative lookahead     ?!= -- negative lookbehind

    phrases = [phrase.strip() for phrase in re.split(pattern, text) if phrase]
    while "" in phrases:
        phrases.remove("")
    return phrases


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """

    if not (my_isinstance(phrases, list) and my_isinstance(stop_words, list)):
        return None

    text_splited = " † ".join(phrases).lower().split()
    stop_words_destroyed = " ".join([word if word not in stop_words else "†" for word in text_splited])
    key_phrases_empty_items = [tuple(phrase.split()) for phrase in re.split("†", stop_words_destroyed)]
    key_phrases = list(filter(lambda x: x != tuple(), key_phrases_empty_items))
    return key_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """

    if not my_isinstance(candidate_keyword_phrases, list):
        return None

    # chain.from_iterable() -- gets chained inputs from a
    # single iterable argument that is evaluated lazily.

    all_words = list(itertools.chain.from_iterable(candidate_keyword_phrases))
    return {word: all_words.count(word) for word in frozenset(all_words)}


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

    if not (my_isinstance(candidate_keyword_phrases, list) or my_isinstance(content_words, list)):
        return None

    word_degree = {word: sum(len(phrase) for phrase in candidate_keyword_phrases
                             if word in phrase) for word in content_words}
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


    if not (is_dic_correct(word_degrees, False, str, int) and
            is_dic_correct(word_frequencies, False, str, int) and
            all(word_degrees.get(word) for word in word_degrees)):
        return None

    word_scores = {word: (word_degrees[word] / word_frequencies[word]) for word in word_degrees}
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

    if not (my_isinstance(candidate_keyword_phrases, list)
            and for_i_empty_checker(candidate_keyword_phrases) and is_dic_correct(word_scores, False, str, float) and
            all(word in word_scores for word in list(itertools.chain.from_iterable(candidate_keyword_phrases)))):
        return None

    cum_score = {phrase: sum(word_scores[word] for word in phrase) for phrase in candidate_keyword_phrases}
    return cum_score


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

    if not (is_dic_correct(keyword_phrases_with_scores, False, tuple, float)
            and my_isinstance(top_n, int) and my_isinstance(max_length, int)
            and top_n >= 1 and max_length >= 1):
        return None

    top_n_sorted = sorted([" ".join(phrase) for phrase in keyword_phrases_with_scores if len(phrase) <= max_length],
                          key=lambda phrase: keyword_phrases_with_scores[tuple(phrase.split())], reverse=True)[:top_n]
    return top_n_sorted


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

    if not (my_isinstance(candidate_keyword_phrases, list) and my_isinstance(phrases, list)):
        return None

    phrases_with_stop_words = []
    pairwised_phrases = tuple(itertools.pairwise(candidate_keyword_phrases))
    #  pairwise('ABCD') => AB BC CD...

    connected_phrases = tuple(pair for pair in frozenset(pairwised_phrases)
                              if pairwised_phrases.count(pair) >= 2)

    for pair, len1, len2 in [(pair, len(pair[0]), len(pair[1])) for pair in connected_phrases]:
        for phrase in [tuple(phrase.lower().split()) for phrase in phrases]:
            for start, stop_word, end in [(i, i + len1, i + len1 + len2) for i in range(len(phrase) - len1 - len2)]:
                if pair == (phrase[start:stop_word], phrase[stop_word + 1:end + 1]):
                    phrases_with_stop_words.append(phrase[start:end + 1])
    phrases_with_adjoining = [phrase for phrase in frozenset(phrases_with_stop_words)
                              if phrases_with_stop_words.count(phrase) >= 2]
    return phrases_with_adjoining


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

    if not (my_isinstance(candidate_keyword_phrases, list) and
            is_dic_correct(word_scores, False, str, float) and
            my_isinstance(stop_words, list)):
        return None

    cumulative_score = {phrase: sum(word_scores[word] for word in phrase
                                    if word not in stop_words)
                        for phrase in candidate_keyword_phrases}
    return cumulative_score


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """

    if not (my_isinstance(text, str) and my_isinstance(max_length, int) and max_length >= 2):
        return None

    pattern = re.compile(r"(?<=^)[^\s\w]+|(?<=\s)[^\s\w]+|[^\s\w]+(?=\s)|[^\s\w]+(?=$)")

    tokens = re.sub(pattern, "", text.lower()).split()
    freqs = {token: tokens.count(token) for token in frozenset(tokens)}
    sorted_dict = sorted(freqs.values(), reverse=True)
    percentile = sorted_dict[int(len(freqs) * 0.20)]

    stop_words_generated = [word for word in sorted(freqs)
                            if freqs[word] >= percentile and
                            len(word) <= max_length]

    return stop_words_generated


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """

    if not my_isinstance(path, Path):
        return None

    with open(path, "r", encoding="utf8") as f:
        dict_stop_words = dict(json.load(f))
    return dict_stop_words


def text_processing(text: str, stop_words: Optional[Sequence[str]] = None, max_length: Optional[int] = None) \
        -> Optional[Mapping[KeyPhrase, float]]:
    """
    Uses all functions written for text processing, key words extracting, etc.
    Arguments: text, stop words list / stop word's maximum length if they need to be generated.
    Returns extracted key phrases in case of correct arguments or None otherwise.
    """
    candidate_keyword_phrases, word_frequencies, word_degrees, word_scores, keyword_phrases_with_scores, \
        candidates_adjoined, cumulative_score_with_stop_words = [None for _ in range(7)]

    phrases = extract_phrases(text)
    if not stop_words and max_length:
        stop_words = generate_stop_words(text, max_length)
    if phrases and stop_words:
        key_phrases = extract_candidate_keyword_phrases(phrases, stop_words)
    if key_phrases:
        word_frequencies = calculate_frequencies_for_content_words(key_phrases)
    if key_phrases and word_frequencies:
        word_degrees = calculate_word_degrees(key_phrases, list(word_frequencies))
    if word_degrees and word_frequencies:
        word_scores = calculate_word_scores(word_degrees, word_frequencies)
    if key_phrases and word_scores:
        keyword_phrases_with_scores = calculate_cumulative_score_for_candidates(key_phrases, word_scores)
    if key_phrases and phrases:
        phrases_with_adjoining = (
            extract_candidate_keyword_phrases_with_adjoining(key_phrases, phrases))
    if phrases_with_adjoining and word_scores and stop_words:
        cumulative_score_with_stop_words = (
            calculate_cumulative_score_for_candidates_with_stop_words(phrases_with_adjoining, word_scores, stop_words))
    else:
        cumulative_score_with_stop_words = {}
    if keyword_phrases_with_scores and cumulative_score_with_stop_words is not None:
        return {**keyword_phrases_with_scores, **cumulative_score_with_stop_words}
    return None
