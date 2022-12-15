"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from typing import Sequence
from lab_2_keywords_cooccurrence.main import (
    extract_phrases, extract_candidate_keyword_phrases,
    calculate_frequencies_for_content_words, calculate_word_degrees,
    calculate_word_scores, calculate_cumulative_score_for_candidates, get_top_n,
    extract_candidate_keyword_phrases_with_adjoining,
    calculate_cumulative_score_for_candidates_with_stop_words, generate_stop_words, load_stop_words)


def read_target_text(file_path: Path) -> str:
    """
    Utility functions that reads the text content from the file
    :param file_path: the path to the file
    :return: the text content of the file
    """
    with open(file_path, 'r', encoding='utf-8') as target_text_file:
        return target_text_file.read()


def text_processing(text: str, stop_word: Sequence[str]) -> None:
    candidate_keyword_phrases = None
    frequencies = None
    word_degrees = None
    word_scores = None
    cumulative_score_for_candidates = None
    candidates_with_adjoining = None

    phrases = extract_phrases(text)

    if phrases and stop_word:
        candidate_keyword_phrases = extract_candidate_keyword_phrases(phrases, stop_word)

    if candidate_keyword_phrases:
        frequencies = calculate_frequencies_for_content_words(candidate_keyword_phrases)

    if candidate_keyword_phrases and frequencies:
        word_degrees = calculate_word_degrees(candidate_keyword_phrases, list(frequencies.keys()))

    if word_degrees and frequencies:
        word_scores = calculate_word_scores(word_degrees, frequencies)

    if candidate_keyword_phrases and word_scores:
        cumulative_score_for_candidates = calculate_cumulative_score_for_candidates(candidate_keyword_phrases,
                                                                                    word_scores)
    if cumulative_score_for_candidates:
        print(get_top_n(cumulative_score_for_candidates, 7, 5))

    if candidate_keyword_phrases and phrases:
        candidates_with_adjoining = extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases,
                                                                                     phrases)
    if stop_word and candidates_with_adjoining and word_scores:
        calculate_cumulative_score_for_candidates_with_stop_words(candidates_with_adjoining, word_scores, stop_word)


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

    RESULT = None

    for exact_text in corpus:
        text_processing(corpus[exact_text], stop_words)

    LOADED_STOP_WORDS = load_stop_words(ASSETS_PATH / 'stopwords.json')

    if LOADED_STOP_WORDS:
        text_processing(read_target_text(ASSETS_PATH / 'polish.txt'), LOADED_STOP_WORDS['pl'])

    UNKNOWN_TEXT = read_target_text(ASSETS_PATH / 'unknown.txt')
    STOP_WORDS_IN_UNKNOWN_TEXT = generate_stop_words(UNKNOWN_TEXT, 10)
    if STOP_WORDS_IN_UNKNOWN_TEXT:
        text_processing(UNKNOWN_TEXT, STOP_WORDS_IN_UNKNOWN_TEXT)

    RESULT = True
    assert RESULT, 'Keywords are not extracted'
