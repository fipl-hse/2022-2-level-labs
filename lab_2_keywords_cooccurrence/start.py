"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from typing import Sequence
from lab_2_keywords_cooccurrence.main import (extract_phrases, extract_candidate_keyword_phrases,
                                              calculate_frequencies_for_content_words, calculate_word_degrees,
                                              calculate_word_scores, calculate_cumulative_score_for_candidates,
                                              get_top_n, extract_candidate_keyword_phrases_with_adjoining,
                                              calculate_cumulative_score_for_candidates_with_stop_words,
                                              load_stop_words, generate_stop_words)


def read_target_text(file_path: Path) -> str:
    """
    Utility functions that reads the text content from the file
    :param file_path: the path to the file
    :return: the text content of the file
    """
    with open(file_path, 'r', encoding='utf-8') as target_text_file:
        return target_text_file.read()


def extract_and_show_keyword_phrases(text: str, stopwords: Sequence[str]) -> None:
    """
    Given a text, prints keyword phrases obtained by different algorithms
    """
    (key_phrases, frequencies, word_degrees, word_scores,
     with_adjoining, cumulative_scores,
     candidates_with_stop_words, stop_word_key_phrases) = [None for _ in range(8)]
    phrases = extract_phrases(text)
    if phrases:
        key_phrases = extract_candidate_keyword_phrases(phrases, stopwords)

    print(key_phrases)

    if key_phrases:
        frequencies = calculate_frequencies_for_content_words(key_phrases)
    if key_phrases and frequencies:
        word_degrees = calculate_word_degrees(key_phrases, list(frequencies.keys()))
    if word_degrees and frequencies:
        word_scores = calculate_word_scores(word_degrees, frequencies)
    if key_phrases and word_scores:
        cumulative_scores = calculate_cumulative_score_for_candidates(key_phrases, word_scores)

    if cumulative_scores:
        print(get_top_n(cumulative_scores, 5, 3))

    if key_phrases and phrases:
        stop_word_key_phrases = extract_candidate_keyword_phrases_with_adjoining(key_phrases, phrases)
    if key_phrases and stop_word_key_phrases:
        with_adjoining = [*key_phrases, *stop_word_key_phrases]
    if with_adjoining and word_scores:
        candidates_with_stop_words \
            = calculate_cumulative_score_for_candidates_with_stop_words(with_adjoining, word_scores, stopwords)

    if candidates_with_stop_words:
        print(get_top_n(candidates_with_stop_words, 5, 3))


if __name__ == "__main__":
    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as fd:
        stop_words = fd.read().split('\n')

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH_GENOME = ASSETS_PATH / 'genome_engineering.txt'
    TARGET_TEXT_PATH_ALBATROSS = ASSETS_PATH / 'albatross.txt'
    TARGET_TEXT_PATH_PAIN_DETECTION = ASSETS_PATH / 'pain_detection.txt'
    TARGET_TEXT_PATH_GAGARIN = ASSETS_PATH / 'gagarin.txt'

    corpus = {
        'gagarin': read_target_text(TARGET_TEXT_PATH_GAGARIN),
        'albatross': read_target_text(TARGET_TEXT_PATH_ALBATROSS),
        'genome_engineering': read_target_text(TARGET_TEXT_PATH_GENOME),
        'pain_detection': read_target_text(TARGET_TEXT_PATH_PAIN_DETECTION)
    }

    for ARTICLE in corpus:
        extract_and_show_keyword_phrases(corpus[ARTICLE], stop_words)

    POLISH_TEXT = read_target_text(ASSETS_PATH / 'polish.txt')
    STOPWORDS_FOR_DIFFERENT_LANGUAGES = load_stop_words(ASSETS_PATH / 'stopwords.json')

    if STOPWORDS_FOR_DIFFERENT_LANGUAGES:
        extract_and_show_keyword_phrases(POLISH_TEXT, STOPWORDS_FOR_DIFFERENT_LANGUAGES['pl'])

    ESPERANTO_TEXT = read_target_text(ASSETS_PATH / 'unknown.txt')
    ESPERANTO_STOPWORDS = generate_stop_words(ESPERANTO_TEXT, 2)
    if ESPERANTO_STOPWORDS:
        extract_and_show_keyword_phrases(ESPERANTO_TEXT, ESPERANTO_STOPWORDS)

    RESULT = "I've made it!"

    assert RESULT, 'Keywords are not extracted'
