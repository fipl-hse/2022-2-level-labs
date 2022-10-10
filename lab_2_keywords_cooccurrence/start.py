"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (extract_phrases,
                                              extract_candidate_keyword_phrases,
                                              calculate_frequencies_for_content_words,
                                              calculate_word_degrees,
                                              calculate_word_scores,
                                              calculate_cumulative_score_for_candidates,
                                              get_top_n,
                                              extract_candidate_keyword_phrases_with_adjoining,
                                              calculate_cumulative_score_for_candidates_with_stop_words)


def read_target_text(file_path: Path) -> str:
    """
    Utility functions that reads the text content from the file
    :param file_path: the path to the file
    :return: the text content of the file
    """
    with open(file_path, 'r', encoding='utf-8') as target_text_file:
        return target_text_file.read()


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
    CANDIDATE_KEYWORD_PHRASES, FREQUENCIES_FOR_CONTENT_WORDS, WORD_DEGREES, WORD_SCORE, CUMULATIVE_SCORE, \
        CANDIDATE_KEYWORD_PHRASES_WITH_ADJOINING = [None for not_def in range(6)]
    PHRASES = extract_phrases(corpus['gagarin'])
    if PHRASES:
        CANDIDATE_KEYWORD_PHRASES = extract_candidate_keyword_phrases(PHRASES, stop_words)

    if CANDIDATE_KEYWORD_PHRASES:
        FREQUENCIES_FOR_CONTENT_WORDS = calculate_frequencies_for_content_words(CANDIDATE_KEYWORD_PHRASES)

    if CANDIDATE_KEYWORD_PHRASES and FREQUENCIES_FOR_CONTENT_WORDS:
        WORD_DEGREES = calculate_word_degrees(CANDIDATE_KEYWORD_PHRASES, list(FREQUENCIES_FOR_CONTENT_WORDS.keys()))

    if WORD_DEGREES and FREQUENCIES_FOR_CONTENT_WORDS:
        WORD_SCORE = calculate_word_scores(WORD_DEGREES, FREQUENCIES_FOR_CONTENT_WORDS)

    if CANDIDATE_KEYWORD_PHRASES and WORD_SCORE:
        CUMULATIVE_SCORE = calculate_cumulative_score_for_candidates(CANDIDATE_KEYWORD_PHRASES, WORD_SCORE)

    if CUMULATIVE_SCORE:
        print(get_top_n(CUMULATIVE_SCORE, 5, 3))

    if CANDIDATE_KEYWORD_PHRASES and PHRASES:
        CANDIDATE_KEYWORD_PHRASES_WITH_ADJOINING = extract_candidate_keyword_phrases_with_adjoining(
            CANDIDATE_KEYWORD_PHRASES, PHRASES)

    if CANDIDATE_KEYWORD_PHRASES_WITH_ADJOINING and WORD_SCORE:
        print(calculate_cumulative_score_for_candidates_with_stop_words(CANDIDATE_KEYWORD_PHRASES_WITH_ADJOINING,
                                                                        WORD_SCORE, stop_words))

    # assert RESULT, 'Keywords are not extracted'
