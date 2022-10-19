"""
Lab 2
Extract keywords based on co-occurrence frequency
"""
from pathlib import Path
from typing import Optional, Sequence, Mapping
import json

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
    punctuation = """,;:¡!¿?…⋯‹›«»/\\"“”[]()⟨⟩}{&|-–~—"""
    for i in text:
        if i in punctuation:
            text = text.replace(i, '.')
    splited_text = text.split('.')
    tokens = []
    for token in splited_text:
        token = token.strip()
        if token:
            tokens.append(token)
    return tokens


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(phrases, list) or not isinstance(stop_words, list) or not stop_words or not phrases:
        return None

    phrases = [i.lower() for i in phrases]
    tokens = []
    keywords = []
    for sentence in phrases:
        sublist = []
        element = ''.join(sentence)
        words = element.split()
        tokens.append(words)
        for word in words:
            if word not in stop_words:
                sublist.append(word)
                continue
            if sublist:
                subtuple = tuple(sublist)
                keywords.append(subtuple)
                sublist = []
        if sublist:
            subtuple = tuple(sublist)
            keywords.append(subtuple)
    return keywords


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(candidate_keyword_phrases, list) or not candidate_keyword_phrases:
        return None
    keywords_list = []
    for keyphrase in candidate_keyword_phrases:
        for word in keyphrase:
            keywords_list.append(word)
    frequencies = {}
    for i in keywords_list:
        frequencies[i] = keywords_list.count(i)
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
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(content_words, list) \
            or not candidate_keyword_phrases or not content_words:
        return None
    wd_dict = {}
    for word in content_words:
        wdegree = 0
        for phrase in candidate_keyword_phrases:
            if word in phrase:
                wdegree += len(phrase)
            else:
                continue
        wd_dict[word] = wdegree
    return wd_dict


def calculate_word_scores(word_degrees: Mapping[str, int],
                          word_frequencies: Mapping[str, int]) -> Optional[Mapping[str, float]]:
    """
    Calculate the word score based on the word degree and word frequency metrics

    :param word_degrees: a mapping between the word and the degree
    :param word_frequencies: a mapping between the word and the frequency
    :return: a dictionary with {word: word_score}

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(word_degrees, dict) or not isinstance(word_frequencies, dict) \
            or not word_degrees or not word_frequencies:
        return None
    wordscore_dict = {}
    for k, value in word_degrees.items():
        if k not in word_frequencies.keys():
            return None
        wordscore_dict[k] = value / word_frequencies[k]
    return wordscore_dict


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
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(word_scores, dict) \
            or not candidate_keyword_phrases or not word_scores:
        return None
    c_score_dict = {}
    for phrase in candidate_keyword_phrases:
        c_score = 0
        for word in phrase:
            if word not in word_scores.keys():
                return None
            c_score += word_scores[word]
        c_score_dict[phrase] = c_score
    return c_score_dict


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
    if not isinstance(keyword_phrases_with_scores, dict) or not isinstance(top_n, int) \
            or not isinstance(max_length, int):
        return None
    if not keyword_phrases_with_scores or not max_length or not top_n or (max_length < 0) or (top_n < 0):
        return None
    top_list = []
    sorted_keys = sorted(keyword_phrases_with_scores, reverse=True, key=lambda word: keyword_phrases_with_scores[word])
    for phrase in sorted_keys:
        if len(phrase) <= max_length:
            phrase = list(phrase)
            phrase = ' '.join(phrase)
            top_list.append(phrase)
    return top_list[:top_n]


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
    if not isinstance(candidate_keyword_phrases, list) or not isinstance(phrases, list) \
            or not candidate_keyword_phrases or not phrases:
        return None

    phrases_list = [' '.join(phrase) for phrase in candidate_keyword_phrases]

    list_of_pairs = [tuple(phrases_list[i:i + 2]) for i in range(len(phrases_list) - 1)]

    pairs_dict = {phrases_pair: list_of_pairs.count(phrases_pair) for phrases_pair in list_of_pairs}

    phrases_tokenized = [i.lower().split(' ') for i in phrases]
    phrases_tokenized = [item for i in phrases_tokenized for item in i if item]

    new_keywords = []
    for k, value in pairs_dict.items():
        if value < 2:
            continue
        token_key = ' '.join(k)
        tokenized_key = token_key.split()

        for ind, word in enumerate(phrases_tokenized):
            list_with_stopword = []
            if word in tokenized_key:
                needed_phrase = phrases_tokenized[ind:ind + (len(tokenized_key) + 1)]
                needed_phrase2 = []
                for i in needed_phrase:
                    word_str = str(i)
                    needed_phrase2.append(word_str)
                list_with_stopword.append(needed_phrase2)

            if not list_with_stopword:
                continue
            flat_list_with_stopword = [item for i in list_with_stopword for item in i if item]
            if flat_list_with_stopword[0] == tokenized_key[0]:
                tuple_with_stopword = tuple(flat_list_with_stopword)
                new_keywords.append(tuple_with_stopword)

    for i in new_keywords.copy():
        if new_keywords.count(i) < 2:
            new_keywords.remove(i)

    new_keywords_final = []
    for i in new_keywords:
        if i not in new_keywords_final:
            new_keywords_final.append(i)

    return new_keywords_final


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
            or not isinstance(stop_words, list):
        return None
    if not candidate_keyword_phrases or not word_scores or not stop_words:
        return None
    c_score_dict = {}
    for phrase in candidate_keyword_phrases:
        c_score = 0
        for word in phrase:
            if word not in stop_words:
                c_score += word_scores[word]
        c_score_dict[phrase] = c_score
    return c_score_dict


def generate_stop_words(text: str, max_length: int) -> Optional[Sequence[str]]:
    """
    Generates the list of stop words from the given text

    :param text: the text
    :param max_length: maximum length (in characters) of an individual stop word
    :return: a list of stop words
    """
    if not isinstance(text, str) or not isinstance(max_length, int) or not text or not max_length or max_length < 0:
        return None
    text = text.replace('\n', ' ').lower()
    punctuation = """,;:¡!¿?…⋯‹›«»/\\"“”[]()⟨⟩}{&|-~—"""
    for i in text:
        if i in punctuation:
            text = text.replace(i, '.')
    splited_text = text.split('.')
    tokens_not_final = [i.lower().split(' ') for i in splited_text]
    tokens = [item for i in tokens_not_final for item in i if item]
    freq_dict = {token: tokens.count(token) for token in tokens}
    percentile = sorted(freq_dict.values())[int((len(freq_dict) / 100 * 70))]
    stop_words = []
    for k, value in freq_dict.items():
        if value >= percentile and len(k) <= max_length:
            stop_words.append(k)

    return stop_words


def load_stop_words(path: Path) -> Optional[Mapping[str, Sequence[str]]]:
    """
    Loads stop word lists from the file
    :param path: path to the file with stop word lists
    :return: a dictionary containing the language names and corresponding stop word lists
    """
    if not isinstance(path, Path) or not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        json_file = dict(json.load(f))
    return json_file
