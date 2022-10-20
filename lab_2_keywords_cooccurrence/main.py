"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any


KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def correct_types(user: Any, expected_type: Any, can_be_empty: bool = False) -> bool:
    """
    Checks type of variable and compares it with expected type.
    For dict and list checks whether their elements are empty (regulated with can_be_empty)
    """
    if not (isinstance(user, expected_type) and user):
        return False
    if expected_type == list:
        for element in user:
            if not element and can_be_empty is False:
                return False
    elif expected_type == dict:
        for key, value in user.items():
            if not (key and value):
                return False
    return True

def check_dict(dictionary: Any, first_type: Any, second_type: Any, empty: bool) -> bool:
    """
    Checks that type of 'dictionary' is dict and verifies the contents of 'dictionary' with the type that
    the user specifies
    """
    if not isinstance(dictionary, dict):
        return False
    if not dictionary and not empty:
        return False
    for key, value in dictionary.items():
        if not isinstance(key, first_type) and not isinstance(value, second_type):
            return False
    return True



def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not (correct_types(text, str) and text):
        return None
    separators = ',;:¡!¿?…⋯‹›«»\\"“”[]()⟨⟩}{&|-–~—'
    for separator in separators:
        text = text.replace(separator, '.')
    split_text = text.split('.')
    new_split_text = []
    for phrase in split_text:
        phrase = phrase.strip()
        if phrase:
            new_split_text.append(phrase)
    return new_split_text

def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not correct_types(phrases, list) or not correct_types(stop_words, list):
        return None
    if not all(correct_types(element, str) for element in phrases):
        return None
    low_phrases = []
    for i in phrases:
        low_phrases.append(i.lower().split())
    for i in range(len(low_phrases)):
        low_phrases[i] = low_phrases[i]
    candidate_keyword_phrases = []
    for one_phrase in low_phrases:
        new_tuple = []
        for words in one_phrase:
            if words not in stop_words:
                new_tuple.append(words)
            elif new_tuple:
                candidate_keyword_phrases.append(tuple(new_tuple))
                new_tuple = []
        if new_tuple:
            candidate_keyword_phrases.append(tuple(new_tuple))
    return candidate_keyword_phrases




def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not correct_types(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    candidate_list = []
    for key_phrase in candidate_keyword_phrases:
        for word in key_phrase:
            candidate_list.append(word)
    corresp_frequencies = {}
    for i in candidate_list:
        corresp_frequencies[i] = candidate_list.count(i)
    return corresp_frequencies


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
    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    if not isinstance(content_words, list) or not content_words:
        return None
    word_degrees = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word in word_degrees and word in content_words:
                word_degrees[word] += len(phrase)
            elif word in content_words:
                word_degrees[word] = len(phrase)
        for word in content_words:
            if word not in word_degrees.keys():
                word_degrees[word] = 0
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
    if not check_dict(word_degrees, str, int, False) or not check_dict(word_frequencies, str, int, False):
        return None
    for word in word_degrees:
        if word not in word_frequencies:
            return None
    return {word: word_degrees[word] / word_frequencies[word] for word in word_degrees if word in word_frequencies}


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
    if (not isinstance(candidate_keyword_phrases, list) or candidate_keyword_phrases == []
            or not isinstance(word_scores, dict) or word_scores == {}):
        return None
    cumulative_dict = {}
    for phrase in candidate_keyword_phrases:
        phrase_score = 0
        for word in phrase:
            if word not in word_scores:
                return None
            else:
                phrase_score += word_scores[word]
        cumulative_dict[phrase] = phrase_score
    return cumulative_dict




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
    if (not isinstance(keyword_phrases_with_scores, dict)
            or keyword_phrases_with_scores == {}
            or not isinstance(top_n, int)
            or top_n <= 0
            or not isinstance(max_length, int)
            or max_length <= 0):
        return None
    small_phrases = {}
    for phrase, score in keyword_phrases_with_scores.items():
        if len(phrase) <= max_length:
            small_phrases[phrase] = score
    sorted_items = sorted(small_phrases.items(), key=lambda x: x[1], reverse=True)
    top_keyword = sorted_items[:top_n]
    simple_phrases = []
    for element in top_keyword:
        simple_phrases.append(' '.join(element[0]))
    return simple_phrases



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
