"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
import re
import json
from math import floor

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_input(user_input: Any, required_type: type) -> bool:
    """
    Checks if the input is as required, and it is not empty (for int, float and str)
    """
    if user_input and isinstance(user_input, required_type):
        return True
    return False


def check_list(user_input: Any, elements_required_type: type) -> bool:
    """
    Checks if the input is not empty, is a sequence and the elements are of the required type
    :param user_input: An input, which is checked
    :param elements_required_type: The type of input's elements
    :return: True if the input is correct and it's elements are of a required type
    """
    if not user_input or not isinstance(user_input, list):
        return False
    for element in user_input:
        if not isinstance(element, elements_required_type):
            return False
    return True


def check_dict(user_input: Any, keys_type: type, values_type: type, values_sec_type: type = dict) -> bool:
    """
    Checks if the input is a non-empty dictionary with required keys and values
    :param user_input: An input, which is checked
    :param keys_type: The type of dictionary's keys
    :param values_type: The type of dictionary's values
    :param values_sec_type: by default=dict (because it won't be used). Optional parameter,
    takes a second value type if it's required
    :return: True if the input is correct and keys and values are as required
    """
    if not user_input or not isinstance(user_input, dict):
        return False
    if values_sec_type:
        for key, value in user_input.items():
            if not isinstance(key, keys_type) or not isinstance(value, (values_type, values_sec_type)):
                return False
    else:
        for key, value in user_input.items():
            if not isinstance(key, keys_type) or not isinstance(value, values_type):
                return False
    return True


def check_keyphrases(user_input: Any) -> bool:
    """
    Checks whether the input is a non-empty keyphrase
    :param user_input: An input, which is checked
    :return: True if the input is correct
    """
    if not user_input or not isinstance(user_input, list):
        return False
    for i in user_input:
        if not isinstance(i, tuple):
            return False
        for j in i:
            if not isinstance(j, str):
                return False
    return True


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not check_input(text, str):
        return None
    punctuat = """!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~¿—–⟨⟩«»…⋯‹›\\¡“”"""
    sep_phrases = text[:]
    for i in sep_phrases:
        if i in punctuat:
            sep_phrases = sep_phrases.replace(i, '*')
    phrases_by_re = re.split(r'[*]+', sep_phrases)
    new_sep = phrases_by_re[:]
    for j in phrases_by_re:
        if re.fullmatch(r'\s+', j) or len(j) == 0:
            new_sep.remove(j)
    result = []
    for i in new_sep:
        result.append(i.strip())
    return result


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not check_list(phrases, str) or not check_list(stop_words, str):
        return None
    lowed_phrases = []
    for phrs in phrases:
        lowed_phrases.append(phrs.lower().split())
    key_candidates = []
    for one_phrase in lowed_phrases:
        future_tuple = []
        for words in one_phrase:
            if words not in stop_words:
                future_tuple.append(words)
            elif future_tuple:
                key_candidates.append(tuple(future_tuple))
                future_tuple = []
        if future_tuple:
            key_candidates.append(tuple(future_tuple))
    return key_candidates


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_keyphrases(candidate_keyword_phrases):
        return None
    freq_dict = {}
    for phrases in candidate_keyword_phrases:
        for words in phrases:
            freq_dict[words] = freq_dict.get(words, 0) + 1
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
    if not check_keyphrases(candidate_keyword_phrases) or not check_list(content_words, str):
        return None
    word_degree_dict = {}
    for one_phrase in candidate_keyword_phrases:
        for words in one_phrase:
            if words in content_words:
                word_degree_dict[words] = word_degree_dict.get(words, 0) + len(one_phrase)
    for i in content_words:
        word_degree_dict[i] = word_degree_dict.get(i, 0)
    return word_degree_dict


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not check_dict(word_degrees, str, int) or not check_dict(word_frequencies, str, int):
        return None
    word_score_dict = {}
    for i in word_degrees:
        if i in word_frequencies:
            word_score_dict[i] = word_degrees[i] / word_frequencies[i]
        else:
            return None
    return word_score_dict


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
    if not check_keyphrases(candidate_keyword_phrases) or not check_dict(word_scores, str, float):
        return None
    cumul_score_dict = {}
    for one_phrase in candidate_keyword_phrases:
        metric = 0.0
        for words in one_phrase:
            if word_scores.get(words, 0) == 0:
                return None
            metric += word_scores[words]
        cumul_score_dict[one_phrase] = metric
    return cumul_score_dict


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
    if (not check_dict(keyword_phrases_with_scores, tuple, float)
            or not check_input(top_n, int) or not check_input(max_length, int) or max_length < 0):
        return None
    if top_n < 1 or max_length < 1:
        return None
    correct_len = {}
    for key, val in keyword_phrases_with_scores.items():
        if len(key) <= max_length:
            correct_len[key] = val
    # sorted_dictionary = sorted(correct_len.keys(), key=correct_len.get, reverse=True)[:top_n]
    sorted_dictionary = [key for (key, value) in sorted(correct_len.items(), key=lambda x: x[1], reverse=True)][:top_n]
    top_phr = []
    for i in sorted_dictionary:
        top_phr.append(' '.join(i))
    return top_phr


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
    if not check_keyphrases(candidate_keyword_phrases) or not check_list(phrases, str):  # checking types
        return None
    possible_phr = {}  # a dictionary for possible phrases, values = frequency of the phrase
    for one_phrase in range(len(candidate_keyword_phrases)-1):
        key_phr = tuple([candidate_keyword_phrases[one_phrase], candidate_keyword_phrases[one_phrase+1]])
        possible_phr[key_phr] = possible_phr.get(key_phr, 0) + 1
    neighbours = []
    for key, val in possible_phr.items():
        if val > 1:
            neighbours.append(list(key))
    all_words = []
    for each_phrase in phrases:
        all_words.append(each_phrase.lower())
    result_list = []
    for each_one in neighbours:
        for sentences in all_words:
            for idx in range(1):
                first_phr = ' '.join(each_one[idx])
                second_phr = ' '.join(each_one[idx+1])
                if first_phr in sentences and second_phr in sentences:
                    result_list.extend(re.findall(r'{}\s\w+\s{}'.format(first_phr, second_phr), sentences))
    freq_dict = {}
    words_list = []
    for one_phrase in result_list:
        words_list.append(tuple(one_phrase.split()))
    for phr_tuple in words_list:
        freq_dict[phr_tuple] = freq_dict.get(phr_tuple, 0) + 1
    return [key for key, value in freq_dict.items() if value > 1]


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
    if (not check_keyphrases(candidate_keyword_phrases) or not check_dict(word_scores, str, float, int)
            or not check_list(stop_words, str)):
        return None
    cumul_score_dict = {}
    for one_phrase in candidate_keyword_phrases:
        metric = 0.0
        for words in one_phrase:
            if words in stop_words:
                metric += 0.0
            elif word_scores.get(words, 0.0):
                metric += word_scores[words]
            else:
                return None
        cumul_score_dict[one_phrase] = metric
    return cumul_score_dict


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    if not check_input(text, str) or not check_input(max_length, int) or max_length < 1:
        return None
    extracted_phr = extract_phrases(text)
    if extracted_phr:
        extracted_words = []
        for phrase in extracted_phr:
            split_phrase = phrase.split()
            for words in split_phrase:
                extracted_words.append(words.lower())
        freq_dict = {}
        for i in extracted_words:
            freq_dict[i] = freq_dict.get(i, 0) + 1
        correct_length_dict = freq_dict.copy()
        for keys in freq_dict:
            if len(keys) > max_length:
                del correct_length_dict[keys]
        # sort_frequency = sorted(correct_length_dict.keys(), key=correct_length_dict.get, reverse=True)
        # sort_frequency = [key for (key, value) in sorted(correct_length_dict.items(), key=lambda x: x[1],
        #                                                reverse=True)]
        sort_frequency = sorted(correct_length_dict.keys(), key=lambda x: correct_length_dict[x], reverse=True)
        sorted_dict = {}
        for i in sort_frequency:
            sorted_dict[i] = correct_length_dict[i]
        if len(sorted_dict) % 10 == 0:
            percent_word = (len(sorted_dict) // 10) * 2 + 1
        else:
            percent_word = floor((len(sorted_dict)/10)*2)
        new_dict = [key for key, value in sorted_dict.items()]
        return new_dict[:percent_word]
    return None


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not isinstance(path, Path) or not path:
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return dict(json.load(file))
