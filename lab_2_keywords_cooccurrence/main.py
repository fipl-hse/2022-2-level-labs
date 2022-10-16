"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
from string import punctuation
from lab_1_keywords_tfidf.main import check_positive_int

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def check_type(user_var: Any, expected_type: Any, can_be_empty: bool = False) -> bool:
    """
    Checks type of variable and compares it with expected type.
    For dict and list checks whether their elements are empty (regulated with can_be_empty)
    """
    if not (isinstance(user_var, expected_type) and user_var):
        return False
    if expected_type == list:
        for element in user_var:
            if not element and can_be_empty is False:
                return False
    elif expected_type == dict:
        for key, value in user_var.items():
            if not (key and value):
                return False
    return True


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str) or not text:
        return None
    for i in punctuation + "¡¿…⋯‹›«»“”⟨⟩–—":
        text = text.replace(i, ",")
    phrases = [phrase.strip() for phrase in text.split(",")]
    phrases = [phrase for phrase in phrases if phrase]
    return phrases


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not check_type(phrases, list) or not check_type(stop_words, list):
        return None
    candidate_keyword_phrases = []
    prepare_candidate_phrases = []
    for phrase in phrases:
        cleaning = []
        for word in phrase.lower().split():
            if word not in stop_words:
                cleaning.append(word)
            else:
                prepare_candidate_phrases.append(cleaning)
                cleaning = []
        prepare_candidate_phrases.append(cleaning)
    for candidate_phrase in prepare_candidate_phrases:
        if candidate_phrase:
            candidate_keyword_phrases.append(tuple(candidate_phrase))
    return candidate_keyword_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_type(candidate_keyword_phrases, list):
        return None
    freq_dict = {word: sum(phrase.count(word) for phrase in candidate_keyword_phrases)
                 for phrase in candidate_keyword_phrases for word in set(phrase)}
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
    if not check_type(candidate_keyword_phrases, list) or not check_type(content_words, list):
        return None
    word_degrees = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word in content_words:
                word_degrees[word] = len(phrase) + word_degrees.get(word, 0)
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
    if not check_type(word_degrees, dict)\
            or not check_type(word_frequencies, dict)\
            or not word_degrees.keys() == word_frequencies.keys():
        return None
    word_scores = {}
    for word in word_degrees.keys():
        if word not in word_frequencies:
            return None
        word_scores[word] = word_degrees[word] / word_frequencies[word]
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
    if not check_type(candidate_keyword_phrases, list) or not check_type(word_scores, dict):
        return None
    keyword_phrases_with_scores = {}
    for phrase in candidate_keyword_phrases:
        score = 0
        for word in phrase:
            if word not in word_scores:
                return None
            score += int(word_scores.get(word))
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
    if not check_type(keyword_phrases_with_scores, dict)\
            or not check_positive_int(top_n)\
            or not check_positive_int(max_length):
        return None
    top_keyword_phrases = []
    top = sorted(keyword_phrases_with_scores.keys(),
                 key=lambda some_phrase: keyword_phrases_with_scores[some_phrase],
                 reverse=True)
    for phrase in top:
        if len(phrase) <= max_length:
            phrase = " ".join(phrase)
            top_keyword_phrases.append(phrase)
    return top_keyword_phrases[:top_n]


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
    if not check_type(candidate_keyword_phrases, list)\
            or not check_type(phrases, list, True):
        return None
    list_of_phrases = [" ".join(phrase) for phrase in candidate_keyword_phrases]
    full_phrases = [tuple((list_of_phrases[index: index + 2])) for index in range(len(list_of_phrases))]
    frequencies = {phrase: full_phrases.count(phrase) for phrase in full_phrases}
    candidates_with_adjoining = []
    for k in frequencies.keys():
        if frequencies[k] < 2:
            continue
        list_of_keys = [" ".join(list(k)).split()]
        for possible_phrase in list_of_keys:
            for phrase in phrases:
                for index in range(len(phrase)):
                    phrase_with_stop_words = phrase.lower().split()[index:index
                    + len(k[0].split()) + len(k[1].split()) + 1]
                    if set(possible_phrase).issubset(phrase_with_stop_words):
                        candidates_with_adjoining.append(tuple(phrase_with_stop_words))
    for phrase in list(set(candidates_with_adjoining)):
        candidates_with_adjoining.remove(phrase)
    return list(set(candidates_with_adjoining))


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
    if not check_type(candidate_keyword_phrases, list)\
            or not check_type(word_scores, dict)\
            or not check_type(stop_words, list):
        return None
    cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score[phrase] = 0
        for word in phrase:
            if word not in stop_words:
                cumulative_score[phrase] += int(word_scores[word])
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
