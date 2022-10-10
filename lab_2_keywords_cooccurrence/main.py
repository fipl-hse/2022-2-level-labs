"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def correct_list(list1: Any, type1: Any, empty: bool) -> bool:
    """
    Checks that type of 'list1' is list and verifies the contents of 'list1' with the type that the user specifies
    """
    if not isinstance(list1, list):
        return False
    if not list1 and not empty:
        return False
    for index in list1:
        if not isinstance(index, type1):
            return False
    return True


def correct_dict(dictionary: Any, type1: Any, type2: Any, empty: bool) -> bool:
    """
    Checks that type of 'dictionary' is dict and verifies the contents of 'dictionary' with the type that
    the user specifies
    """
    if not isinstance(dictionary, dict):
        return False
    if not dictionary and not empty:
        return False
    for key, value in dictionary.items():
        if not isinstance(key, type1) or not isinstance(value, (int, type2)) or isinstance(value, bool):
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
    punctuation_marks = '''.,;':¡!¿?…⋯‹›«»\\/"“”[]()⟨⟩}{&|-–~—'''
    for mark in punctuation_marks:
        text = text.replace(mark, ',')
    split_text = text.split(',')
    return [phrase.strip() for phrase in split_text if phrase.strip() != '']


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not correct_list(phrases, str, False) or not correct_list(stop_words, str, False):
        return None
    candidates_list = []
    for phrase in phrases:
        phrase = phrase.lower().split()
        candidate1 = []
        for word in phrase:
            if word in stop_words:
                if candidate1:
                    candidate2 = tuple(candidate1)
                    candidates_list.append(candidate2)
                    candidate1.clear()
            elif word == phrase[len(phrase) - 1]:
                candidate1.append(word)
                candidate2 = tuple(candidate1)
                candidates_list.append(candidate2)
                candidate1.clear()
            else:
                candidate1.append(word)
    return candidates_list


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not correct_list(candidate_keyword_phrases, tuple, False):
        return None
    frequencies_for_content_words = {}
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            frequencies_for_content_words[word] = frequencies_for_content_words.get(word, 0) + 1
    return frequencies_for_content_words


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
    if not correct_list(candidate_keyword_phrases, tuple, False) or not correct_list(content_words, str, False):
        return None
    word_degrees = {}
    for phrase in candidate_keyword_phrases:
        for word in content_words:
            if word in phrase:
                word_degrees[word] = word_degrees.get(word, 0) + len(phrase)
            elif word not in word_degrees:
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
    if not correct_dict(word_degrees, str, int, False) or not correct_dict(word_frequencies, str, int, False):
        return None
    for word in word_degrees:
        if word not in word_frequencies:
            return None
    return {word: word_degrees[word]/word_frequencies[word] for word in word_degrees if word in word_frequencies}


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
    if not correct_list(candidate_keyword_phrases, tuple, False) or not correct_dict(word_scores, str, float, False):
        return None
    cumulative_score_for_candidates = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score = 0
        for word in phrase:
            if word not in word_scores:
                return None
            cumulative_score += word_scores[word]
        cumulative_score_for_candidates[phrase] = cumulative_score
    return cumulative_score_for_candidates


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
    if not correct_dict(keyword_phrases_with_scores, tuple, float, False) or not isinstance(top_n, int) \
            or not isinstance(max_length, int) or not max_length > 0 or not top_n > 0:
        return None
    top_phrases = sorted(keyword_phrases_with_scores.keys(), key=lambda word: keyword_phrases_with_scores[word],
                         reverse=True)
    return [" ".join(phrase) for phrase in top_phrases if len(phrase) <= max_length][:top_n]


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
    if not correct_list(candidate_keyword_phrases, tuple, False) or not correct_list(phrases, str, False):
        return None
    count = 0
    count = sum([count+1 for keyword_phrase in candidate_keyword_phrases for item in candidate_keyword_phrases
                 if keyword_phrase == item])
    dict_key = {item: count for item in candidate_keyword_phrases if count >= 2}
    list_of_tuples = [tuple(candidate_keyword_phrases[count:count + 2]) for count, item in
                      enumerate(candidate_keyword_phrases) for key in dict_key.keys() if key == item]
    final_list = []
    for item in list_of_tuples:
        count = 0
        for tuple1 in list_of_tuples:
            if tuple1 == item:
                count += 1
        if count >= 2:
            final_list.append(item)
    result = list(set(final_list))[::-1]

    key_phrases = []
    for item in phrases:
        for elem in result:
            first_tuple, second_tuple = elem[0][0], elem[1][len(elem[1]) - 1]
            if first_tuple in item:
                key_phrases.append(item[item.index(first_tuple):item.rindex(second_tuple) +
                                                                           (len(second_tuple) - 1) + 1])

    if not key_phrases:
        return []
    count2 = 0
    previous = key_phrases[0][0]
    for item in key_phrases:
        for phrase in key_phrases:
            if phrase == item:
                count2 += 1
    list_result = {tuple(item.split()) for item in key_phrases if count2 >= 2 and previous == item[0]}
    return list(list_result)


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
    if not correct_list(candidate_keyword_phrases, tuple, False) or not correct_dict(word_scores, str, float, False) \
            or not correct_list(stop_words, str, False):
        return None
    result_dict = {}
    for item in candidate_keyword_phrases:
        count = 0
        for elem in item:
            for key, value in word_scores.items():
                if elem == key:
                    count += int(value)
                    for stop_word in stop_words:
                        if elem == stop_word:
                            count -= int(value)
        result_dict[item] = count
    return result_dict


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
