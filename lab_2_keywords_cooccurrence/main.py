"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping, Union
from itertools import pairwise
from lab_1_keywords_tfidf.main import check_list, check_dict

KeyPhrase = tuple[str, ...]
KeyPhrases = Sequence[KeyPhrase]


def extract_phrases(text: str) -> Optional[Sequence[str]]:
    """
    Splits the text into separate phrases using phrase delimiters
    :param text: an original text
    :return: a list of phrases

    In case of corrupt input arguments, None is returned
    @rtype: object
    """
    if not isinstance(text, str) or not text:
        return None
    punctuation = '''!"#$%&'()*+,-./:;<=>?@[]^_`{|}~–—¡¿⟨⟩«»'…⋯‹›\\\\“”\\"\\""'''
    for i in punctuation:
        text = text.replace(i, ",")
    split_text = text.split(',')
    phrases = []
    for phrase in split_text:
        phrase = phrase.strip()
        if phrase:
            phrases.append(phrase)
    return phrases


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not check_list(phrases, str, False) or not check_list(stop_words, str, False):
        return None
    candidate_keyword_phrases = []
    for phrase in phrases:
        phrase = phrase.lower().split()
        prepared_key_phrases = []
        for one_word in phrase:
            if one_word not in stop_words:
                prepared_key_phrases.append(one_word)
            elif prepared_key_phrases and one_word in stop_words:
                candidate_keyword_phrases.append(tuple(prepared_key_phrases))
                prepared_key_phrases = []
        if prepared_key_phrases:
            candidate_keyword_phrases.append(tuple(prepared_key_phrases))
    return candidate_keyword_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_list(candidate_keyword_phrases, tuple, False) or not candidate_keyword_phrases:
        return None
    dictionary_freqs = {}
    for one_phrase in candidate_keyword_phrases:
        for one_word in one_phrase:
            dictionary_freqs[one_word] = dictionary_freqs.get(one_word, 0) + 1
    return dictionary_freqs


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
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_list(content_words, str, False):
        return None
    dict_word_degrees = {}
    for word in content_words:
        dict_word_degrees[word] = 0
        for phrase in candidate_keyword_phrases:
            if word in phrase:
                dict_word_degrees[word] += len(phrase)
    return dict_word_degrees


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not (check_dict(word_degrees, str, int, False) and check_dict(word_frequencies, str, int, False)
            and word_degrees.keys() == word_frequencies.keys()):
        return None
    word_score_dict = {}
    for key, value in word_degrees.items():
        word_score_dict[key] = value / word_frequencies.get(key, 1)
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
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_dict(word_scores, str, float, False):
        return None
    cumulative_score_dict = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score = 0
        for word in phrase:
            if word in word_scores:
                cumulative_score += word_scores.get(word, 0)
            else:
                return None
        cumulative_score_dict[phrase] = cumulative_score
    return cumulative_score_dict


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
    if not check_dict(keyword_phrases_with_scores, tuple, float, False) or not isinstance(top_n, int) \
            or not isinstance(max_length, int) or not top_n > 0 or not max_length > 0:
        return None
    list_phrases = [word for (word, value) in sorted(keyword_phrases_with_scores.items(), key=lambda val: val[1],
                                                     reverse=True)]
    if not keyword_phrases_with_scores:
        return None
    if not isinstance(keyword_phrases_with_scores, dict):
        return None
    top_phrases = []
    for phrase in list_phrases:
        if len(phrase) <= max_length:
            phrase = ' '.join(phrase)
            top_phrases.append(phrase)
    return top_phrases[:top_n]


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
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_list(phrases, str, False):
        return None
    list_of_phrases = []
    for phrase in candidate_keyword_phrases:
        list_of_phrases.append(' '.join(phrase))

    phrases_pairs = list(pairwise(list_of_phrases))

    tokens_with_stopword = []
    for phrase in phrases:
        tokens_with_stopword.append(phrase.split(' '))
    tokens_with_stopword = [word.lower() for phrase in tokens_with_stopword for word in phrase]

    dict_tuples = {}
    for phrase in phrases_pairs:
        value = phrases_pairs.count(phrase)
        dict_tuples[phrase] = value

    prepared_adjoining_phrases = []
    for key, value in dict_tuples.items():
        if value < 2:
            continue
        tokens = ' '.join(list(key)).split()
        for idx, k in enumerate(tokens_with_stopword):
            list_stopwords = []
            if k in tokens:
                list_stopwords.append(tokens_with_stopword[idx:idx + len(tokens) + 1])
            if not list_stopwords:
                continue
            list_stopwords = [key for lst in list_stopwords for key in lst if key]
            if list_stopwords[0] == tokens[0]:
                prepared_adjoining_phrases.append(tuple(list_stopwords))

    for some_phrase in set(prepared_adjoining_phrases):
        prepared_adjoining_phrases.remove(some_phrase)
    return list(set(prepared_adjoining_phrases))


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
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_list(stop_words, str, False) \
            or not check_dict(word_scores, str, Union[int, float], False):
        return None
    cumulative_scores_new_phrases = {}
    for phrase in candidate_keyword_phrases:
        cumulative_scores_new_phrases[phrase] = 0
        for one_word in phrase:
            if one_word not in stop_words:
                cumulative_scores_new_phrases[phrase] += int(word_scores[one_word])
    return cumulative_scores_new_phrases


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
