"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (
    extract_phrases,
    extract_candidate_keyword_phrases,
    calculate_frequencies_for_content_words,
    calculate_word_degrees,
    calculate_word_scores,
    calculate_cumulative_score_for_candidates,
    get_top_n,
    extract_candidate_keyword_phrases_with_adjoining,
    calculate_cumulative_score_for_candidates_with_stop_words
)

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

    PHRASES = extract_phrases(corpus['gagarin'])
    TOP = 5
    MAX_LENGTH = 3
    CUMULATIVE_SCORE_FOR_CANDIDATES_WITH_STOP_WORDS = None
    CANDIDATE_KEYWORD_PHRASES_WITH_ADJOINING = None

    if PHRASES:
        CANDIDATE_KEYWORD_PHRASES = extract_candidate_keyword_phrases(PHRASES, stop_words)

    if CANDIDATE_KEYWORD_PHRASES:
        WORD_FREQUENCIES = calculate_frequencies_for_content_words(CANDIDATE_KEYWORD_PHRASES)

    if WORD_FREQUENCIES and CANDIDATE_KEYWORD_PHRASES:
        WORD_DEGREES = calculate_word_degrees(CANDIDATE_KEYWORD_PHRASES, list(WORD_FREQUENCIES.keys()))

    if WORD_FREQUENCIES and WORD_DEGREES:
        WORD_SCORES = calculate_word_scores(WORD_DEGREES, WORD_FREQUENCIES)

    if WORD_SCORES and CANDIDATE_KEYWORD_PHRASES:
        KEYWORD_PHRASES_WITH_SCORES =\
            calculate_cumulative_score_for_candidates(CANDIDATE_KEYWORD_PHRASES, WORD_SCORES)

    if KEYWORD_PHRASES_WITH_SCORES and TOP and MAX_LENGTH:
        GET_TOP_N = get_top_n(KEYWORD_PHRASES_WITH_SCORES, TOP, MAX_LENGTH)

    if CANDIDATE_KEYWORD_PHRASES and PHRASES:
        CANDIDATE_KEYWORD_PHRASES_WITH_ADJOINING =\
            extract_candidate_keyword_phrases_with_adjoining(CANDIDATE_KEYWORD_PHRASES, PHRASES)

    if CANDIDATE_KEYWORD_PHRASES_WITH_ADJOINING and WORD_SCORES:
        CUMULATIVE_SCORE_FOR_CANDIDATES_WITH_STOP_WORDS =\
            calculate_cumulative_score_for_candidates_with_stop_words(CANDIDATE_KEYWORD_PHRASES_WITH_ADJOINING,
                                                                      WORD_SCORES, stop_words)

    RESULT = CUMULATIVE_SCORE_FOR_CANDIDATES_WITH_STOP_WORDS

    assert RESULT, 'Keywords are not extracted'
