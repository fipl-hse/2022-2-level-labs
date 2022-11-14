"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path

from typing import Optional, Sequence, Mapping, Any
from string import punctuation

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str) or not text:
        return None
    full_punctuation = punctuation + "–—¡¿⟨⟩«»…⋯‹›“”"
    for symbol in full_punctuation:
        text = text.replace(symbol, '.')
    phrases = text.split('.')
    phrases_stripped = []
    for string in phrases:
        string = string.strip()
        if string:
            phrases_stripped.append(string)

    return phrases_stripped


def check_list_types(sequence: Sequence, expected_type: Any) -> bool:
    """
    Checks element's type in a list
    :param sequence: a list
    :param expected_type: expected type
    :return: True if element's type is same as expected type, otherwise False
    """
    for item in sequence:
        if not isinstance(item, expected_type):
            return False
    return True


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    check_exist = phrases and stop_words
    check_types = isinstance(phrases, list) and isinstance(stop_words, list)
    if not check_exist or not check_types:
        return None

    check_in_list_types = check_list_types(phrases, str) and check_list_types(stop_words, str)
    if not check_in_list_types:
        return None

    tokens_list = []
    candidate_keywords = []

    for phrase in phrases:
        tokens_list.append(phrase.lower().split())

    for tokens in tokens_list:
        new_candidate = []
        for token in tokens:
            if token not in stop_words:
                new_candidate.append(token)
            else:
                if new_candidate:
                    candidate_keywords.append(tuple(new_candidate))
                    new_candidate.clear()
        if new_candidate:
            candidate_keywords.append(tuple(new_candidate))
    return candidate_keywords


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not candidate_keyword_phrases:
        return None
    if not isinstance(candidate_keyword_phrases, list):
        return None
    if not check_list_types(candidate_keyword_phrases, tuple):
        return None
    for sequence in candidate_keyword_phrases:
        if not check_list_types(sequence, str):
            return None

    frequency_dict = {}
    for item in candidate_keyword_phrases:
        for token in item:
            if token in frequency_dict:
                frequency_dict[token] += 1
            else:
                frequency_dict[token] = 1
    return frequency_dict


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
    if not candidate_keyword_phrases or not content_words:
        return None
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(content_words, list):
        return None
    if not check_list_types(candidate_keyword_phrases, tuple) or not check_list_types(content_words, str):
        return None
    for sequence in candidate_keyword_phrases:
        if not check_list_types(sequence, str):
            return None

    word_degrees_dict = {}
    for word in content_words:
        word_degrees_dict[word] = 0

    for item in candidate_keyword_phrases:
        for token in item:
            if token in content_words:
                word_degrees_dict[token] += len(item)
    return word_degrees_dict


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not word_degrees or not word_frequencies:
        return None
    if not isinstance(word_degrees, dict) or not isinstance(word_frequencies, dict):
        return None

    word_score = {}
    for word in word_degrees:
        if word in word_frequencies:
            word_score[word] = word_degrees[word] / word_frequencies[word]
        else:
            return None
    return word_score


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
    if not candidate_keyword_phrases or not word_scores:
        return None
    if not isinstance(word_scores, dict) or not isinstance(candidate_keyword_phrases, list):
        return None
    if not check_list_types(candidate_keyword_phrases, tuple):
        return None

    phrases_scores_dict = {}
    for phrase in candidate_keyword_phrases:
        phrases_scores_dict[phrase] = 0
        for word in phrase:
            if word in word_scores:
                phrases_scores_dict[phrase] += word_scores[word]
            else:
                return None
    return phrases_scores_dict


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
    if not isinstance(keyword_phrases_with_scores, dict) or not isinstance(top_n, int) or not isinstance(max_length,
                                                                                                         int):
        return None
    if not keyword_phrases_with_scores or max_length <= 0 or top_n <= 0:
        return None

    filtered_phrases = []
    for phrase in keyword_phrases_with_scores:
        if len(phrase) <= max_length:
            filtered_phrases.append(phrase)
    sorted_phrases = sorted(filtered_phrases, key=lambda phrase: keyword_phrases_with_scores[phrase], reverse=True)

    joined_phrases = []
    for phrase in sorted_phrases:
        joined_phrases.append(' '.join(phrase))
    return joined_phrases[:top_n]


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
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(phrases, list):
        return None
    if not candidate_keyword_phrases or not phrases:
        return None

    pairs = [(candidate_keyword_phrases[i], candidate_keyword_phrases[i + 1])
             for i in range(len(candidate_keyword_phrases) - 1)]

    repeated_pairs = [pair for pair in set(pairs) if pairs.count(pair) > 1]

    keyword_phrases_with_stopwords = []
    for pair in repeated_pairs:
        pair_1, pair_2 = (' '.join(element) for element in pair)
        for phrase in phrases:
            phrase = phrase.lower()
            pair_2_end = 0
            while (pair_1_i := phrase.find(pair_1, pair_2_end)) != -1 and (
                    pair_2_i := phrase.find(pair_2, pair_2_end)) != -1:
                pair_1_end = pair_1_i + len(pair_1)
                pair_2_end = pair_2_i + len(pair_2)
                stopword = (phrase[pair_1_end: pair_2_i].strip(),)
                keyword_phrases_with_stopwords.append(pair[0] + stopword + pair[1])

    repeated_keyword_phrases = []
    for keyword_phrase in keyword_phrases_with_stopwords:
        if keyword_phrases_with_stopwords.count(keyword_phrase) > 1 and keyword_phrase not in repeated_keyword_phrases:
            repeated_keyword_phrases.append(keyword_phrase)
    return repeated_keyword_phrases


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
    if not candidate_keyword_phrases or not word_scores or not stop_words:
        return None
    if not isinstance(word_scores, dict) or not isinstance(candidate_keyword_phrases, list) or not isinstance(
            stop_words, list):
        return None
    if not check_list_types(candidate_keyword_phrases, tuple) or not check_list_types(stop_words, str):
        return None

    phrases_scores_dict = {}
    for phrase in candidate_keyword_phrases:
        phrases_scores_dict[phrase] = sum([word_scores[word] for word in phrase
                                           if word in word_scores and word not in stop_words])
    return phrases_scores_dict


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
