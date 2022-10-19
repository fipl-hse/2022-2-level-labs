"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping

from typing import Any
import re
from lab_1_keywords_tfidf.main import check_list

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]

def check_dict(user_input: Any, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    """
    Checks weather object is dictionary
    hat has keys and values of certain type
    """
    if not isinstance(user_input, dict):
        return False
    if not user_input and can_be_empty is False:
        return False
    for key, value in user_input.items():
        if not (isinstance(key, key_type) and isinstance(value, value_type)):
            return False
    return True

def check_key_phrase(arg_to_check: Any) -> bool:
    """
    Check weather object has a KeyPhrase type (list of tuples of strings)
    """
    if not (arg_to_check and isinstance(arg_to_check, list)):
        return False
    for element in arg_to_check:
        if not isinstance(element, tuple):
            return False
        for word in element:
            if not isinstance(word, str):
                return False
    return True

def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not (text and isinstance(text, str)):
        return None
    list_of_phrases = re.split(r"[^\w ]+", text)
    for phrase in list_of_phrases:
        list_of_phrases[list_of_phrases.index(phrase)] = phrase.strip()
    list_of_phrases_copy = [phrase for phrase in list_of_phrases if phrase != '']
    return list_of_phrases_copy

def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if check_list(phrases, str, False) and check_list(stop_words, str, False):
        phrases_lower = [phrase.lower() for phrase in phrases]
        list_of_phrases = [phrase.split() for phrase in phrases_lower]
        for index, phrase in enumerate(list_of_phrases):
            for idx, word in enumerate(phrase):
                if word in stop_words:
                    list_of_phrases[index][idx] = '00'
            list_of_phrases[index].append('00')
        new_list = []
        counter = []
        for index, phrase in enumerate(list_of_phrases):
            for idx, word in enumerate(phrase):
                if word == '00':
                    counter.extend(word + ' ')
                    new_list.append(''.join(counter))
                    counter.clear()
                else:
                    counter.extend(word + ' ')
            while '00 ' in new_list:
                new_list.remove('00 ')
            final_list = [string.strip('00 ') for string in new_list]
            tfinal_list = [tuple(phrase1.split()) for phrase1 in final_list]
        return tfinal_list
    return None


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if check_key_phrase(candidate_keyword_phrases):
        words_in_text = []
        dict_of_keywords = {}
        for elem in candidate_keyword_phrases:
            words_in_text.extend(elem)
            for word in elem:
                dict_of_keywords[word] = words_in_text.count(word)
        return dict_of_keywords
    return None

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
    if check_key_phrase(candidate_keyword_phrases) and check_list(content_words, str, False):
        new_dict = {word: 0 for word in content_words}
        for elem in candidate_keyword_phrases:
            for word in content_words:
                if word in elem:
                    new_dict[word] += len(elem)
        return new_dict
    return None


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if check_dict(word_degrees, str, int, False) and check_dict(word_frequencies, str, int, False):
        word_scores = {}
        for word, degree in word_degrees.items():
            try:
                word_scores[word] = degree / word_frequencies[word]
            except KeyError:
                return None
        return word_scores
    return None

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
    if check_key_phrase(candidate_keyword_phrases) and check_dict(word_scores, str, float, False):
        set_of_candidates = set(candidate_keyword_phrases)
        phrases_scores = {}
        for phrase in set_of_candidates:
            for word in phrase:
                if word in word_scores:
                    phrases_scores[phrase] = 0.0
                else:
                    return None
        for phrase in set_of_candidates:
            for word in phrase:
                phrases_scores[phrase] += word_scores[word]
        return phrases_scores
    return None

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
    if not (keyword_phrases_with_scores and isinstance(keyword_phrases_with_scores, dict) and isinstance(top_n,
        int) and isinstance(max_length, int) and max_length > 0 and top_n > 0):
        return None
    for key1, value1 in keyword_phrases_with_scores.items():
        if not (isinstance(key1, tuple) and isinstance(value1, float)):
            return None
        for elem in key1:
            if not isinstance(elem, str):
                return None
    list_of_kw = keyword_phrases_with_scores.keys()
    top_n_phrases = sorted(list_of_kw, key=lambda k: keyword_phrases_with_scores[k], reverse=True)
    key_phrases = [' '.join(phrase) for phrase in top_n_phrases if len(phrase) <= max_length]
    return key_phrases[:top_n]

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
    if check_key_phrase(candidate_keyword_phrases) and check_list(phrases, str, False):
        list_of_phrases = [phrase.split() for phrase in phrases]
        phrases_lower = [phrase.lower() for phrase in phrases]
        key_phrases_with_stop_words = []
        for index1, phrase1 in enumerate(list_of_phrases):
            for phrase in candidate_keyword_phrases:
                if phrase[-1] in list_of_phrases[index1]:
                    phrase_index1 = list_of_phrases[index1].index(phrase[-1])
                    words_lst = []
                    for word in phrase1[phrase_index1 + 2:]:
                        words_lst.append(word)
                        twords_lst = tuple(words_lst)
                        if twords_lst in candidate_keyword_phrases:
                            left_phrase = ' '.join(list(phrase))
                            right_phrase = ' '.join(words_lst)
                            pair_of_phrases = re.findall(left_phrase+r" \w+ "+right_phrase, ' '.join(phrases_lower))
                            key_phrases = [tuple(key_phrase.split()) for key_phrase
                                    in pair_of_phrases if
                                    len(set(pair_of_phrases)) == 1 and len(pair_of_phrases) >= 2]
                            key_phrases_with_stop_words.extend(key_phrases)
        key_phrases_with_stop_words_set = set(key_phrases_with_stop_words)
        return list(key_phrases_with_stop_words_set)
    return None

def calculate_cumulative_score_for_candidates_with_stop_words(candidate_keyword_phrases: KeyPhrases,
                                                              word_scores: Mapping[str, float],
                                                              stop_words: Sequence[str]) \
        -> Optional[Mapping[KeyPhrase, float]]:
    """
    Calculate cumulative score for each candidate keyword phrase. Cumulative score for a keyword phrase equals to
    the sum of the phrase scores of each keyword phrase's constituent except for the stop words

    :param candidate_keyword_phrases: a list of candidate keyword phrases
    :param word_scores: phrase scores
    :param stop_words: a list of stop words
    :return: a dictionary containing the mapping between the candidate keyword phrases and respective cumulative scores

    In case of corrupt input arguments, None is returned
    """
    if check_key_phrase(candidate_keyword_phrases) and isinstance(word_scores, dict)\
            and word_scores and check_list(stop_words, str, False):
        set_of_candidates = set(candidate_keyword_phrases)
        phrases_scores = {phrase: 0.0 for phrase in set_of_candidates}
        for phrase in set_of_candidates:
            for word in phrase:
                if word not in stop_words:
                    phrases_scores[phrase] += word_scores[word]
                else:
                    continue
        return phrases_scores
    return None

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
