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
    if not (isinstance(text, str) and text):
        return None
    new_string = ''
    for sym in text:
        if not (sym.isalnum() or sym.isspace()):
            sym = ','
        new_string += sym
    phrases = new_string.split(',')
    phrases = list(filter(None, phrases))
    phrases[:] = [x for x in phrases if x.strip()]
    for phrase in phrases:
        phrases[phrases.index(phrase)] = phrase.strip()
    return phrases


def check_content(massive, type_name) -> Optional[bool]:
    """
    Checks if all elements in a sequence is the same type

    Parameters:
    massive: A sequence to check
    type_name: name of type (str, int, etc)

    Returns:
    True

    In case of different types of elements, None is returned
    """
    if not (massive and all(isinstance(el, type_name) for el in massive)):
        return None
    return True


def check_list(lst: list[str], stop_words: list[str]) -> list[str]:
    """
    Checks if there are stop-words in list and delete them

    :param lst: a list of words to check
    :param stop_words: a list of the stop words
    :return: list without stop-words
    """
    for el in lst:
        if el in stop_words:
            while el in lst:
                lst.remove(el)
    return lst


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(phrases, list) and isinstance(stop_words, list) and
            check_content(phrases, str) and check_content(stop_words, str)):
        return None
    candidate = []
    for phrase in phrases:
        idx_lst = []
        split_phrase = phrase.lower().split()
        for idx, token in enumerate(split_phrase):
            if token in stop_words:
                idx_lst.append(idx)
        if len(idx_lst) == len(split_phrase):
            return candidate
        if not idx_lst:
            candidate.append(tuple(split_phrase))
        for i in range(len(idx_lst)):
            if len(idx_lst) == 1:
                new_phrase = check_list(split_phrase[:idx_lst[i]], stop_words)
                candidate.append(tuple(new_phrase))
            if i == len(idx_lst) - 1:
                new_phrase = check_list(split_phrase[idx_lst[i] + 1:], stop_words)
            else:
                new_phrase = check_list(split_phrase[idx_lst[i] + 1: idx_lst[i + 1]], stop_words)
            candidate.append(tuple(new_phrase))
    return list(filter(None, candidate))


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(candidate_keyword_phrases, list) and candidate_keyword_phrases):
        return None
    word_freqs = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            if word not in word_freqs:
                word_freqs[word] = 1
            else:
                word_freqs[word] = word_freqs[word] + 1
    return word_freqs


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
    if not (candidate_keyword_phrases and isinstance(candidate_keyword_phrases, Sequence) and content_words and
            isinstance(content_words, Sequence)):
        return None
    w_degree = {}
    for word in content_words:
        if not isinstance(word, str):
            return None
        w_degree[word] = 0
        for el in candidate_keyword_phrases:
            if word in el:
                w_degree[word] = w_degree[word] + len(el)
    return w_degree


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not (word_frequencies and word_degrees and isinstance(word_degrees, Mapping) and
            isinstance(word_frequencies, Mapping)):
        return None
    if word_frequencies.keys() != word_degrees.keys():
        return None
    word_scores = {}
    for word in word_frequencies:
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
    if not (isinstance(candidate_keyword_phrases, list) and candidate_keyword_phrases and word_scores
            and isinstance(word_scores, Mapping)):
        return None
    phrase_scores = {}
    for el in candidate_keyword_phrases:
        phrase_scores[el] = 0
        for word in el:
            if word not in word_scores:
                return None
            phrase_scores[el] = phrase_scores[el] + word_scores[word]
    return phrase_scores


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
    if not (keyword_phrases_with_scores and isinstance(keyword_phrases_with_scores, Mapping) and isinstance(top_n, int)
            and isinstance(max_length, int) and max_length > 0 and top_n > 0):
        return None
    new_dict = {}
    for k, v in keyword_phrases_with_scores.items():
        if len(k) <= max_length:
            new_dict[k] = v
    top_list = sorted(new_dict.keys(), key=lambda key: new_dict[key], reverse=True)[:top_n]
    print(top_list)
    top_n_list = []
    for el in top_list:
        top_n_list.append(' '.join(list(el)))
    return top_n_list


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
