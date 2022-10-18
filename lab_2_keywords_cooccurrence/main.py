"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping

import re

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
    if not (phrases and stop_words and isinstance(phrases, list) and isinstance(stop_words, list)):
        return None
    for element1 in phrases:
        if not isinstance(element1, str):
            return None
    for element2 in stop_words:
        if not isinstance(element2, str):
            return None
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
        final_list = [tuple(phrase1.split()) for index1, phrase1 in enumerate(final_list)]
    return final_list


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not (candidate_keyword_phrases and isinstance(candidate_keyword_phrases, list)):
        return None
    for element in candidate_keyword_phrases:
        if not isinstance(element, tuple):
            return None
        for word in element:
            if not isinstance(word, str):
                return None
    words_in_text = []
    dict_of_keywords = {}
    for elem in candidate_keyword_phrases:
        words_in_text.extend(elem)
        for word in elem:
            dict_of_keywords[word] = words_in_text.count(word)
    return dict_of_keywords

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
    if not (candidate_keyword_phrases and content_words and isinstance(candidate_keyword_phrases, list) and isinstance(
            content_words, Sequence)):
        return None
    for words in content_words:
        if not isinstance(words, str):
            return None
    for phrase in candidate_keyword_phrases:
        if not isinstance(phrase, tuple):
            return None
        for word_1 in phrase:
            if not isinstance(word_1, str):
                return None
    new_dict = {word: 0 for word in content_words}
    for elem in candidate_keyword_phrases:
        for word in content_words:
            if word in elem:
                new_dict[word] += len(elem)
    return new_dict


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not (word_frequencies and word_degrees and isinstance(word_frequencies, Mapping) and isinstance(word_degrees,
            Mapping)):
        return None
    for key1, value1 in word_degrees.items():
        if not (isinstance(key1, str) and isinstance(value1, int)):
            return None
    for key2, value2 in word_frequencies.items():
        if not (isinstance(key2, str) and isinstance(value2, int)):
            return None
    word_scores = {}
    for word, degree in word_degrees.items():
        try:
            word_scores[word] = degree / word_frequencies[word]
        except KeyError:
            return None
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
    if not (candidate_keyword_phrases and word_scores and isinstance(candidate_keyword_phrases,
        list) and isinstance(word_scores, Mapping)):
        return None
    for elem in candidate_keyword_phrases:
        if not isinstance(elem, tuple):
            return None
        for wrd in elem:
            if not isinstance(wrd, str):
                return None
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
    if not (keyword_phrases_with_scores and isinstance(keyword_phrases_with_scores, Mapping) and isinstance(top_n,
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
    if not (candidate_keyword_phrases and phrases and isinstance(candidate_keyword_phrases,
        list) and isinstance(phrases, Sequence)):
        return None
    for words in phrases:
        if not isinstance(words, str):
            return None
    for phrase0 in candidate_keyword_phrases:
        if not isinstance(phrase0, tuple):
            return None
        for word_1 in phrase0:
            if not isinstance(word_1, str):
                return None
    list_of_phrases = [phrase.split() for phrase in phrases]
    phrases_lower = [phrase.lower() for phrase in phrases]
    key_phrases_with_stop_words = []
    for phrase in candidate_keyword_phrases:
        for i, j in enumerate(list_of_phrases):
            if phrase[-1] in list_of_phrases[i]:
                phrase_index1 = list_of_phrases[i].index(phrase[-1])
                words_lst = []
                for cccc in j[phrase_index1 + 2:]:
                    words_lst.append(cccc)
                    twords_lst = tuple(words_lst)
                    if twords_lst in candidate_keyword_phrases:
                        u = ' '.join(list(phrase))
                        o = ' '.join(words_lst)
                        pair_of_phrases = re.findall(u+r" \w+ "+o, ' '.join(phrases_lower))
                        pair_of_phrases_ext = []
                        for i, c in enumerate(pair_of_phrases):
                            if i == 0:
                                pair_of_phrases_ext.append(c)
                            else:
                                if c == pair_of_phrases_ext[i-1]:
                                    pair_of_phrases_ext.append(c)
                        count_pairs = len(pair_of_phrases_ext)
                        if count_pairs >= 2:
                            key_phrase = tuple(phrase + tuple(j[phrase_index1+1].split()) + twords_lst)
                            if key_phrase not in key_phrases_with_stop_words:
                                key_phrases_with_stop_words.append(key_phrase)
    return key_phrases_with_stop_words

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
    if not (candidate_keyword_phrases and word_scores and stop_words and isinstance(candidate_keyword_phrases,
        list) and isinstance(word_scores, dict) and isinstance(stop_words, list)):
        return None
    for elem in candidate_keyword_phrases:
        if not isinstance(elem, tuple):
            return None
        for wrd in elem:
            if not isinstance(wrd, str):
                return None
    for key_1, value_2 in word_scores.items():
        if not (isinstance(key_1, str) and (isinstance(value_2, float) or isinstance(value_2, int))):
            return None
    for element2 in stop_words:
        if not isinstance(element2, str):
            return None
    set_of_candidates = set(candidate_keyword_phrases)
    phrases_scores = {phrase: 0.0 for phrase in set_of_candidates}
    for phrase in set_of_candidates:
        for word in phrase:
            if word not in stop_words:
                phrases_scores[phrase] += word_scores[word]
            else:
                continue
    return phrases_scores

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
