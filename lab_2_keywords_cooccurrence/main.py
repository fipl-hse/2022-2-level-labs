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
    punctuation = '''.,;':¡!¿?…⋯‹›«»\\/"“”[]()⟨⟩}{&|-–~—'''
    for punc in punctuation:
        text = text.replace(punc, ',')
    phrase_list = text.split(',')
    return [new_phrase for phrase in phrase_list if (new_phrase := phrase.strip())]


def extract_candidate_keyword_phrases(phrases: Sequence[str], stop_words: Sequence[str]) -> Optional[KeyPhrases]:
    """
    Creates a list of candidate keyword phrases by splitting the given phrases by the stop words
    :param phrases: a list of the phrases
    :param stop_words: a list of the stop words
    :return: the candidate keyword phrases for the text

    In case of corrupt input arguments, None is returned
    """
    if not (phrases and isinstance(phrases, list) and stop_words and isinstance(stop_words, list)):
        return None
    for phrase in phrases:
        if not isinstance(phrase, str):
            return None
    phrase_lst = []
    tuple_lst = []
    candidate_keyword_phrases = []
    for phrase in phrases:
        phrase = phrase.lower()
        phrase_lst.append(phrase.split())
    for phrase in phrase_lst:
        for word in phrase:
            if word not in stop_words:
                tuple_lst.append(word)
            else:
                phrase_tpl = tuple(tuple_lst)
                if phrase_tpl:
                    candidate_keyword_phrases.append(phrase_tpl)
                tuple_lst.clear()
        if tuple_lst:
            candidate_keyword_phrases.append(tuple(tuple_lst))
            tuple_lst.clear()
    return candidate_keyword_phrases


def calculate_frequencies_for_content_words(candidate_keyword_phrases: KeyPhrases) -> Optional[Mapping[str, int]]:
    """
    Extracts the content words from the candidate keyword phrases list and computes their frequencies
    :param candidate_keyword_phrases: a list of the candidate keyword phrases
    :return: a dictionary with the content words and corresponding frequencies

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(candidate_keyword_phrases, list) and candidate_keyword_phrases):
        return None
    frequencies_dict = {}
    for phrase in candidate_keyword_phrases:
        for token in phrase:
            frequencies_dict[token] = frequencies_dict.get(token, 0) + 1
    return frequencies_dict


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
    if not (isinstance(word_degrees, dict) and word_degrees):
        return None
    if not (isinstance(word_frequencies, dict) and word_frequencies):
        return None
    word_scores = {}
    for word in word_degrees:
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
    if not (isinstance(candidate_keyword_phrases, list) and candidate_keyword_phrases):
        return None
    if not (isinstance(word_scores, dict) and word_scores):
        return None
    cumulative_score_for_candidates = {}
    for phrase in candidate_keyword_phrases:
        phrase_score = 0.0
        for word in phrase:
            if word not in word_scores:
                return None
            phrase_score += word_scores[word]
        cumulative_score_for_candidates[phrase] = phrase_score
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
    if not (isinstance(keyword_phrases_with_scores, dict) and keyword_phrases_with_scores):
        return None
    if not (isinstance(top_n, int) and top_n > 0):
        return None
    if not (isinstance(max_length, int) and max_length > 0):
        return None

    top = sorted(keyword_phrases_with_scores.keys(), key=lambda key: keyword_phrases_with_scores[key], reverse=True)
    top_phrases = []
    for phrase in top:
        if len(phrase) <= max_length:
            top_phrases.append(' '.join(phrase))
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
    if not (candidate_keyword_phrases and isinstance(candidate_keyword_phrases, list)
            and phrases and isinstance(phrases, list)):
        return None
    keyword_phrases_with_adj = []

    for keyword_phrase, phrase in zip(candidate_keyword_phrases, phrases):
        splited_phrase = phrase.split()
        keyword_phrase_freq = candidate_keyword_phrases.count(keyword_phrase)
        next_phrase = candidate_keyword_phrases[candidate_keyword_phrases.index(keyword_phrase) + 1]
        next_phrase_freq = candidate_keyword_phrases.count(next_phrase)

        for keyword, word in zip(keyword_phrase, next_phrase):
            if keyword in splited_phrase and word in splited_phrase and keyword_phrase_freq > 1 and next_phrase_freq > 1:
                next_phrase_start_idx = splited_phrase.index(next_phrase[0])
                stop_word = splited_phrase[next_phrase_start_idx - 1]
                word_idx = next_phrase.index(word)
                keyword_phrases_with_adj.append(tuple([keyword] + [stop_word] + list(next_phrase[word_idx:])))
    return keyword_phrases_with_adj


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
    if not(isinstance(candidate_keyword_phrases, list) and candidate_keyword_phrases and isinstance(word_scores, dict)
           and word_scores and isinstance(stop_words, list) and stop_words):
        return None
    cumulative_scores = {}
    for phrase in candidate_keyword_phrases:
        phrase_score = 0.0
        for word in phrase:
            if word not in stop_words:
                if word not in word_scores.keys():
                    return None
                phrase_score += word_scores[word]
                cumulative_scores[phrase] = phrase_score

    return cumulative_scores


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
