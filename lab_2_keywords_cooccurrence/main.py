"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping

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
    punctuation = '''.,;:¡!¿?…⋯‹›«»"“”\\[]()⟨⟩}{&]|[-–~—]'''
    for symbol in text:
        if symbol in punctuation:
            text = text.replace(symbol, ',')
    phrases_list = text.split(',')
    return [phrase.strip() for phrase in phrases_list if phrase.strip()]


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(phrases, list):
        return None
    if not (isinstance(stop_words, list)) or isinstance(stop_words, bool):
        return None
    for phrase in phrases:
        if not isinstance(phrase, str) or not phrase:
            return None

    if not phrases or not stop_words:
        return None

    for word in stop_words:
        if not isinstance(word, str):
            return None


    candidate_phrases = []
    for phrase in phrases:
        phrase = phrase.lower().split()
        lst = []
        for word in phrase:
            if word not in stop_words:
                lst.append(word)
            elif (word in stop_words) and lst:
                candidate_phrases.append(tuple(lst))
                lst.clear()
        if lst:
            candidate_phrases.append(tuple(lst))
            lst.clear
    return candidate_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not (candidate_keyword_phrases and isinstance(candidate_keyword_phrases, list)):
        return None

    content_words = []
    for phrase in candidate_keyword_phrases:
        content_words.append(list(phrase))
    content_words = sum(content_words, start=[])
    return {word: content_words.count(word) for word in content_words}


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
    if not (candidate_keyword_phrases and content_words):
        return None
    if not (isinstance(candidate_keyword_phrases, list) and isinstance(content_words,list)):
        return None
    degrees = {}
    for word in content_words:
        degrees[word] = sum(len(phrase) for phrase in candidate_keyword_phrases if word in phrase)
    return degrees


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not (word_degrees and word_frequencies):
        return None
    if not (isinstance(word_degrees, dict) and isinstance(word_frequencies, dict)):
        return None

    for key in word_frequencies.keys():
        if key not in word_degrees.keys():
            return None

    return ({word: word_degrees[word] / word_frequencies[word] for word in word_frequencies.keys()
              if word in word_degrees.keys()})


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
    if not (word_scores and isinstance(word_scores, dict)):
        return None
    if not (candidate_keyword_phrases and isinstance(candidate_keyword_phrases, list)):
        return None

    lst = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score = 0
        for content_word in phrase:
            if content_word not in word_scores:
                return None
            if content_word in word_scores:
                cumulative_score += word_scores[content_word]
        lst[phrase] = cumulative_score
    return lst




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
    if not isinstance(keyword_phrases_with_scores, dict) or not keyword_phrases_with_scores:
        return None

    if top_n <= 0 or not isinstance(top_n, int):
        return None

    if max_length <= 0 or not isinstance(max_length, int):
        return None

    lst = sorted(keyword_phrases_with_scores, reverse=True)

    return lst[:top_n]


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