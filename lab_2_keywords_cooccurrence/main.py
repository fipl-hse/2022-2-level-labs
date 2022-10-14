"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
import re
import json
from copy import deepcopy

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_input(user_input: Any, required_type: type) -> bool:
    """
    Checks if the input is as required, and it is not empty (for int, float and str)
    """
    if user_input and isinstance(user_input, required_type):
        return True


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


def check_dict(user_input: Any, keys_type: type, values_type: type) -> bool:
    """
    Checks if the input is a non-empty dictionary with required keys and values
    :param user_input: An input, which is checked
    :param keys_type: The type of dictionary's keys
    :param values_type: The type of dictionary's values
    :return: True if the input is correct and keys and values are as required
    """
    if not user_input or not isinstance(user_input, dict):
        return False
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
    punctuat = """!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~¿—–⟨⟩«»…⋯‹›\\¡“”"""
    sep_phrases = text[:]
    for i in sep_phrases:
        if i in punctuat:
            sep_phrases = sep_phrases.replace(i, '*')
    sep_phrases = re.split(r'[*]{1,}', sep_phrases)
    new_sep = sep_phrases[:]
    for j in sep_phrases:
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
    lowed_phrases = []  # put lower phrases
    for i in phrases:
        lowed_phrases.append(i.lower())  # putting lower phrases
    for i in range(len(lowed_phrases)):
        lowed_phrases[i] = lowed_phrases[i].split()  # splitting lowed phrases by ' '. until here okay
    key_candidates = []
    for one_phrase in lowed_phrases:  # going through each phrase
        future_tuple = []  # here will be materials for a tuple
        for words in one_phrase:  # going through each word
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
        metric = 0
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
    if not check_dict(keyword_phrases_with_scores, tuple, float) or not check_input(top_n, int) or not check_input(max_length, int) or max_length < 0:
        return None
    if top_n < 1 or max_length < 1:
        return None
    correct_len = {}
    for key, val in keyword_phrases_with_scores.items():
        if len(key) <= max_length:
            correct_len[key] = val
    correct_len = sorted(correct_len.keys(), key=lambda key: correct_len[key], reverse=True)[:top_n]
    top_phr = []
    for i in correct_len:
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
    if not check_keyphrases(candidate_keyword_phrases) or not check_list(phrases, str): # checking types
        return None
    possible_phr = {}  # a dictionary for possible phrases, values = frequency of the phrase
    for one_phrase in range(0, len(candidate_keyword_phrases)-1):
        for sec_phrase in range(one_phrase + 1, len(candidate_keyword_phrases)):
            key_phr = tuple([candidate_keyword_phrases[one_phrase], candidate_keyword_phrases[sec_phrase]])
            possible_phr[key_phr] = possible_phr.get(key_phr, 0) + 1
            break
    neighbours = []
    for key, val in possible_phr.items():
        if val > 1:
            neighbours.append(list(key))
    all_words = []
    for one_phrase in phrases:
        all_words.append(one_phrase.lower().split())  # turning phrases into words
    result_list = deepcopy(neighbours)
    for one_phrase in range(len(neighbours)):
        for words in range(len(neighbours[one_phrase])-1):
            for sentences in all_words:
                first_phr = ' '.join(neighbours[one_phrase][words])
                second_phr = ' '.join(neighbours[one_phrase][words+1])
                str_sentence = ' '.join(sentences)
                if first_phr in str_sentence and second_phr in str_sentence:
                    counter_first = str_sentence.count(first_phr)
                    counter_sec = str_sentence.count(second_phr)
                    if counter_first == 1 and counter_sec == 1:
                        stop_word = sentences.index(first_phr)+1 #index of stop_w
                        if len(result_list[one_phrase]) == 2:
                            result_list[one_phrase].insert(1, (sentences[stop_word],))
                        else:
                            a = neighbours[one_phrase][:]
                            a.insert(1, (sentences[stop_word],))
                            result_list.append(a)
                    elif counter_first == counter_sec and counter_first > 1:
                        for i in range(len(sentences)):
                            if sentences[i] == first_phr:
                                stop_word = sentences.index(first_phr, i)+1 #index of stop_w
                                if len(result_list[one_phrase]) == 2:
                                    result_list[one_phrase].insert(1, (sentences[stop_word],))
                                else:
                                    a = neighbours[one_phrase][:]
                                    a.insert(1, (sentences[stop_word],))
                                    result_list.append(a)
    freq_dict = {}
    for one_phrase in result_list:
        list_for_str = []
        for words in one_phrase:
            list_for_str.append(' '.join(words))
        key_phrase = tuple((' '.join(list_for_str)).split())
        freq_dict[key_phrase] = freq_dict.get(key_phrase, 0) + 1
    return [j for j, i in freq_dict.items() if i > 1]



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
    if not check_keyphrases(candidate_keyword_phrases) or not check_dict(word_scores, str, float | int) or not check_list(stop_words, str):
        return None
    cumul_score_dict = {}
    for one_phrase in candidate_keyword_phrases:
        metric = 0
        for words in one_phrase:
            if words in stop_words:
                metric += 0
            elif word_scores.get(words, 0) != 0:
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
    if not check_input(text, str) or not check_input(max_length, int):
        return None
    extracted_phr = extract_phrases(text)
    if extracted_phr:
        extracted_words = []
        for phrase in extracted_phr:
            phrase = phrase.split()
            for words in phrase:
                extracted_words.append(words.lower())
    if extracted_words:
        for word in extracted_words:
            if len(word) > max_length:
                extracted_words.remove(word)
        frequencies_words = calculate_frequencies_for_content_words(extracted_words)
    if frequencies_words:
        #sort_frequency = sorted(correct_len.keys(), key=lambda key: correct_len[key], reverse=True)
        sort_frequency = sorted(frequencies_words, key=frequencies_words.get)
        sorted_dict = {}
        for i in sort_frequency:
            sorted_dict[i] = frequencies_words[i]
        print(sorted_dict.items())
    # используй калькюлэйт фрикуенси
print(generate_stop_words('cats and, cats cats dogs dogs aRE animals: domestic', 6))

def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not isinstance(path, Path) or not path:
        return None
    with open(path, 'r', encoding='utf-8') as file:
        result_dict = json.load(file)
    return result_dict

def work_with_text(text: str, stop_words: list[str]):
    """
    Calling all the needed functions to process a text:
    extracting phrases, candidate keyword phrases, frequencies for
    content words, word degrees
    """
    if not check_input(text, str) or not check_list(stop_words, str):
        return None
    extracted_phr = extract_phrases(text)
    candidate_keyword_phr = extract_candidate_keyword_phrases(extracted_phr)
    frequencies_content = calculate_frequencies_for_content_words(candidate_keyword_phr)

