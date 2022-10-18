"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping
import re
from lab_1_keywords_tfidf.main import (check_list, check_dict, check_positive_int)


KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]

def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not (text and isinstance(text, str)):
        return None
    punctuation = """!"#$%&'()*+,-./:;<=>?@[]^_`{|}~¿—–⟨⟩«»…⋯‹›\\¡“”"""
    for mark in punctuation:
        text = text.replace(mark, ',')
    cleaned_text = text.split(',')
    new_cleaned_text = [i.strip() for i in cleaned_text if i and not i.isspace()]
    return new_cleaned_text


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
    phrase_list = []
    list_for_tuples = []
    candidate_keywords_phrases = []
    for phrase in phrases:
        phrase = phrase.lower()
        phrase_list.append(phrase.split())
    for phrase in phrase_list:
        for word in phrase:
            if word not in stop_words:
                list_for_tuples.append(word)
            else:
                phrase_tuple = tuple(list_for_tuples)
                if phrase_tuple:
                    candidate_keywords_phrases.append(phrase_tuple)
                list_for_tuples.clear()
        if list_for_tuples:
            candidate_keywords_phrases.append(tuple(list_for_tuples))
            list_for_tuples.clear()
    return candidate_keywords_phrases




def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_list(candidate_keyword_phrases, tuple, False):
        return None
    tokens_list = [word for phrase in candidate_keyword_phrases for word in phrase]
    frequencies = {word: tokens_list.count(word) for word in tokens_list}
    return frequencies



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
    word_degrees = {}
    for word in content_words:
        word_degrees[word] = 0
        for phrase in candidate_keyword_phrases:
            if word in phrase:
                word_degrees[word] += len(phrase)
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
    if not (check_dict(word_degrees, str, int, False) and check_dict(word_frequencies, str, int, False)):
        return None
    word_scores = {}
    for key in word_degrees:
        if key not in word_frequencies.keys():
            return None
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
    if not (check_list(candidate_keyword_phrases, tuple, False) and check_dict(word_scores, str, float, False)):
        return None
    keyword_phrases_with_scores = {}
    for phrase in candidate_keyword_phrases:
        score = 0
        for word in phrase:
            if word not in word_scores.keys():
                return None
            score += word_scores[word]
        keyword_phrases_with_scores[phrase] = score
    return keyword_phrases_with_scores


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
    if not (check_dict(keyword_phrases_with_scores, tuple, float, False) and check_positive_int(top_n)
            and check_positive_int(max_length)):
        return None
    appropriate_phrases = {}
    for phrase, score in keyword_phrases_with_scores.items():
        if len(phrase) <= max_length:
            appropriate_phrases[' '.join(phrase)] = score
    top_score = [key for (key, value) in sorted(appropriate_phrases.items(),
                                                key=lambda val: val[1], reverse=True)][:top_n]
    return top_score

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
    if not (check_list(candidate_keyword_phrases, tuple, False) and check_list(phrases, str, False)):
        return None
    possible_pairs = {}
    for i in range(len(candidate_keyword_phrases) - 1):
        pair = candidate_keyword_phrases[i], candidate_keyword_phrases[i + 1]
        if pair not in possible_pairs:
            possible_pairs[pair] = 1
        else:
            possible_pairs[pair] += 1
    appropriate_pairs = []
    for pair in possible_pairs:
        if possible_pairs.get(pair) > 1:
            appropriate_pairs.append([' '.join(element) for element in pair])
    phrases_with_stopwords = []
    for pair in appropriate_pairs:
        counter = 0
        for phrase in phrases:
            if (pair[0] and pair[1]) in phrase:
                counter += 1
                if extract_phrases(phrase) == [phrase]:
                    phrases_with_stopwords.extend(re.findall(rf'{pair[0]}\s[а-я]+\s{pair[1]}', phrase))
        if counter == 0:
            return []
    new_phrases_with_sw = [phrase.split() for phrase in set(phrases_with_stopwords)]
    final_phrases = [tuple(phrase) for phrase in new_phrases_with_sw]
    return final_phrases

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
    if not (check_list(candidate_keyword_phrases, tuple, False) and check_list(stop_words, str, False)):
        return None
    if not (word_scores and isinstance(word_scores, dict)):
        return None
    for key, value in word_scores.items():
        if not (isinstance(key, str) and isinstance(value, float)):
            if isinstance(value, int):
                value = float(value)
            else:
                return None
    cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        kw_phrase = []
        score = 0
        for word in phrase:
            kw_phrase.append(word)
            if word not in stop_words:
                score += word_scores.get(word)
        cumulative_score[tuple(kw_phrase)] = score
    return cumulative_score



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
