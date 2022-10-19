"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping
from string import punctuation
from lab_1_keywords_tfidf.main import check_list, check_dict

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
    for i in punctuation + '–—];:¡¿⟨⟩&]«»…⋯‹›“”' + '\n':
        text = text.replace(i, ".")
    split_phrases = text.split(".")
    list_of_phrases = []
    for phrase in split_phrases:
        phrase = phrase.strip()
        if phrase:
            list_of_phrases.append(phrase)
    return list_of_phrases


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
    keyword_phrases = []
    tuples_candidate_phrases = []
    for phrase in phrases:
        phrase = phrase.lower().split()
        no_stop_words = []
        for word in phrase:
            if word not in stop_words:
                no_stop_words.append(word)
            else:
                tuples_candidate_phrases.append(no_stop_words)
                no_stop_words = []
        tuples_candidate_phrases.append(no_stop_words)
    for phrase in tuples_candidate_phrases:
        if phrase:
            keyword_phrases.append(tuple(phrase))
    return keyword_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not check_list(candidate_keyword_phrases, tuple, False):
        return None
    tokens = []
    for phrase in candidate_keyword_phrases:
        for word in phrase:
            tokens.append(word)
    frequencies = {}
    for word in tokens:
        frequencies[word] = tokens.count(word)
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
    if not check_list(candidate_keyword_phrases, tuple, False) or not check_list(content_words, str, False):
        return None
    word_degree = {}
    for word in content_words:
        word_degree[word] = 0
        for phrase in candidate_keyword_phrases:
            if word in phrase:
                word_degree[word] += len(phrase)
    return word_degree


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
    key_scores = {}
    for key in word_degrees.keys():
        if key in word_frequencies:
            key_scores[key] = word_degrees[key] / word_frequencies[key]
    return key_scores


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
    cumulative_score_for_candidates = {}
    for phrase in candidate_keyword_phrases:
        score = 0
        for word in phrase:
            if word not in word_scores:
                return None
            score += word_scores.get(word)
        cumulative_score_for_candidates[phrase] = score
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
    if not check_dict(keyword_phrases_with_scores, tuple, float, False) or not isinstance(top_n, int) \
            or not isinstance(max_length, int) or not top_n > 0 or not max_length > 0:
        return None
    true_phrases = [word for (word, value) in sorted(keyword_phrases_with_scores.items(), key=lambda val: val[1],
                                                     reverse=True)]
    top_true_phrases = []
    for phrase in true_phrases:
        if len(phrase) <= max_length:
            phrase = ' '.join(phrase)
            top_true_phrases.append(phrase)
    return top_true_phrases[:top_n]


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

    phrases_list = [' '.join(phrase) for phrase in candidate_keyword_phrases]

    pairs_of_phrases = []
    for phrase in range(len(phrases_list)):
        tuples_of_pairs = tuple((phrases_list[phrase: phrase + 2]))
        pairs_of_phrases.append(tuples_of_pairs)

    tokens_phrases = []
    for phrase in phrases:
        phrase = phrase.split(' ')
        tokens_phrases.append(phrase)
    tokens_phrases = [word.lower() for phrase in tokens_phrases for word in phrase]

    pairs_freq_dict = {phrase: pairs_of_phrases.count(phrase) for phrase in pairs_of_phrases}

    candidates_with_adjoining = []
    for word, value in pairs_freq_dict.items():
        if value < 2:
            continue
        joining_key = ' '.join(word)
        tokens_keys = joining_key.split()
        for ele, word in enumerate(tokens_phrases):
            stop_words = []
            if word in tokens_keys:
                stop_words.append(tokens_phrases[ele:ele + len(tokens_keys) + 1])
            if not stop_words:
                continue
            stop_words = [word for lst in stop_words for word in lst if word]
            if tokens_keys[0] == stop_words[0]:
                candidates_with_adjoining.append(tuple(stop_words))

    for phrase in set(candidates_with_adjoining):
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
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(word_scores, dict) \
            or not isinstance(stop_words, list) or not candidate_keyword_phrases or not word_scores or not stop_words:
        return None
    cumulative_score = {}
    for phrase in candidate_keyword_phrases:
        cumulative_score[phrase] = 0
        for word in word_scores:
            if word in phrase and word not in stop_words:
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
