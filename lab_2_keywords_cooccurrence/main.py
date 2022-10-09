"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from string import punctuation
from typing import Optional, Sequence, Mapping, Any

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_list(user_input: Any, element_type: type, can_be_empty: bool) -> bool:
    """
    Checks weather object is list
    that contains objects of certain type
    """
    if not isinstance(user_input, list):
        return False
    if not user_input and can_be_empty is False:
        return False
    for element in user_input:
        if not isinstance(element, element_type):
            return False
    return True


def check_dict (user_input:Any, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    if not isinstance(user_input, dict):
        return False
    if not user_input and can_be_empty is False:
        return False
    for key, value in user_input.items():
        if not (isinstance(key, key_type) and isinstance(value, value_type)):
            return False
    return True


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(text, str) and text):
        return None
    more_punctuation = punctuation + '–—];:¡¿⟨⟩&]«»…⋯‹›“”' + '\n'
    for i in more_punctuation:
        text = text.replace(i, '!')
    text_list = text.split('!')
    list0 = [strings.strip() for strings in text_list if strings]
    final_list = [string for string in list0 if string != '']
    return final_list


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not (check_list(phrases, str, False) and check_list(stop_words, str, False)):
        return None
    no_stop_words_list = []
    candidate_phrases = []
    for phrase in phrases:
        new_phrase = phrase.lower().split()
        no_stop_words_list_0 = []
        for word in new_phrase:
            if word not in stop_words:
                no_stop_words_list_0.append(word)
            else:
                no_stop_words_list.append(no_stop_words_list_0)
                no_stop_words_list_0 = []
        no_stop_words_list.append(no_stop_words_list_0)
        candidate_phrases = [tuple(candidate) for candidate in no_stop_words_list if candidate]
    return candidate_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_list(candidate_keyword_phrases, tuple, False):
        return None
    words_list = []
    freq_dict = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            words_list.append(word)
    for word in words_list:
        freq_dict[word] = freq_dict.get(word, 0) + 1
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
    if not (check_list(candidate_keyword_phrases, tuple, False) and check_list(content_words, str, False)):
        return None
    degrees_dict = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word in content_words:
                degrees_dict[word] = degrees_dict.get(word, 0) + len(phrase)
        for word in content_words:
            if word not in degrees_dict.keys():
                degrees_dict[word] = 0
    return degrees_dict


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not (check_dict(word_degrees, str, int, False) and check_dict(word_frequencies, str, int, False)
            and word_degrees.keys() == word_frequencies.keys()):
        return None
    word_scores = {}
    for key in word_degrees.keys():
        word_scores[key] = word_degrees[key] / word_frequencies[key]
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
