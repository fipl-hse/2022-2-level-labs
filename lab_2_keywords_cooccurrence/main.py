"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
import re
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
    return bool(instance and not isinstance(instance, bool) and isinstance(instance, int))


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


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not my_isinstance(text, str):
        return None
    punct = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~–—«»“”"
    pattern = re.compile(rf"[{punct}]+(?=[$\s])|(?!=\w)[{punct}]+")
    #  ?= -- positive lookahead
    #  ?!= -- negative lookahead
    return [phrase.strip() for phrase in re.split(pattern, text) if phrase]


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
    pass


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
    pass


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
