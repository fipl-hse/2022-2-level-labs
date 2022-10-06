"""
Co-occurrence-driven keyword extraction starter
"""
from pathlib import Path
from lab_2_keywords_cooccurrence.main import (extract_phrases, extract_candidate_keyword_phrases,
                                              calculate_frequencies_for_content_words, calculate_word_degrees,
                                              calculate_word_scores, calculate_cumulative_score_for_candidates,
                                              get_top_n, extract_candidate_keyword_phrases_with_adjoining,
                                              calculate_cumulative_score_for_candidates_with_stop_words,
                                              generate_stop_words, load_stop_words)


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

    EXTRACT_PHRASES = None
    CANDIDATE_KEYWORD_PHRASES = None
    FREQUENCIES = None
    WORD_DEGREE = None
    WORD_SCORE = None
    SCORE_FOR_CANDIDATES = None
    KEYWORDS_PHRASES_WITH_ADJOINING = None

    for key in corpus:

        EXTRACT_PHRASES = extract_phrases(corpus[key])

        if EXTRACT_PHRASES:
            print(CANDIDATE_KEYWORD_PHRASES := extract_candidate_keyword_phrases(EXTRACT_PHRASES, stop_words))

        if CANDIDATE_KEYWORD_PHRASES:
            FREQUENCIES = calculate_frequencies_for_content_words(CANDIDATE_KEYWORD_PHRASES)

        if FREQUENCIES and CANDIDATE_KEYWORD_PHRASES:
            WORD_DEGREE = calculate_word_degrees(CANDIDATE_KEYWORD_PHRASES, list(FREQUENCIES.keys()))

        if WORD_DEGREE and FREQUENCIES:
            WORD_SCORE = calculate_word_scores(WORD_DEGREE, FREQUENCIES)

        if WORD_SCORE and CANDIDATE_KEYWORD_PHRASES:
            SCORE_FOR_CANDIDATES = calculate_cumulative_score_for_candidates(CANDIDATE_KEYWORD_PHRASES, WORD_SCORE)

        if SCORE_FOR_CANDIDATES:
            print(top_n := get_top_n(SCORE_FOR_CANDIDATES, 10, 10))

        if CANDIDATE_KEYWORD_PHRASES and EXTRACT_PHRASES:
            KEYWORDS_PHRASES_WITH_ADJOINING = extract_candidate_keyword_phrases_with_adjoining(
                CANDIDATE_KEYWORD_PHRASES, EXTRACT_PHRASES)

        if KEYWORDS_PHRASES_WITH_ADJOINING and WORD_SCORE:
            print((score_for_candidates_with_stop_words :=
                   calculate_cumulative_score_for_candidates_with_stop_words(KEYWORDS_PHRASES_WITH_ADJOINING,
                                                                             WORD_SCORE, stop_words)))
    # polish + unknown

    EXTRACT_PHRASES = None
    CANDIDATE_KEYWORD_PHRASES = None
    FREQUENCIES = None
    WORD_DEGREE = None
    WORD_SCORE = None
    SCORE_FOR_CANDIDATES = None
    KEYWORDS_PHRASES_WITH_ADJOINING = None

    TEXT_UNKNOWN = read_target_text(ASSETS_PATH / 'unknown.txt')

    TEXTS = [read_target_text(ASSETS_PATH / 'polish.txt'), TEXT_UNKNOWN]
    STOP_WORDS = [load_stop_words(ASSETS_PATH / 'stopwords.json')["pl"], generate_stop_words(TEXT_UNKNOWN, 25)]

    for ind in range(2):
        EXTRACT_PHRASES = extract_phrases(TEXTS[ind])

        if EXTRACT_PHRASES:
            CANDIDATE_KEYWORD_PHRASES = extract_candidate_keyword_phrases(EXTRACT_PHRASES, STOP_WORDS[ind])

        if CANDIDATE_KEYWORD_PHRASES:
            FREQUENCIES = calculate_frequencies_for_content_words(CANDIDATE_KEYWORD_PHRASES)

        if FREQUENCIES and CANDIDATE_KEYWORD_PHRASES:
            WORD_DEGREE = calculate_word_degrees(CANDIDATE_KEYWORD_PHRASES, list(FREQUENCIES.keys()))

        if WORD_DEGREE and FREQUENCIES:
            WORD_SCORE = calculate_word_scores(WORD_DEGREE, FREQUENCIES)

        if WORD_SCORE and CANDIDATE_KEYWORD_PHRASES:
            SCORE_FOR_CANDIDATES = calculate_cumulative_score_for_candidates(CANDIDATE_KEYWORD_PHRASES, WORD_SCORE)

        if SCORE_FOR_CANDIDATES:
            print(top_n := get_top_n(SCORE_FOR_CANDIDATES, 10, 15))

        if CANDIDATE_KEYWORD_PHRASES and EXTRACT_PHRASES:
            KEYWORDS_PHRASES_WITH_ADJOINING = extract_candidate_keyword_phrases_with_adjoining(
                CANDIDATE_KEYWORD_PHRASES, EXTRACT_PHRASES)

        if KEYWORDS_PHRASES_WITH_ADJOINING and WORD_SCORE:
            print((score_for_candidates_with_stop_words :=
                   calculate_cumulative_score_for_candidates_with_stop_words(KEYWORDS_PHRASES_WITH_ADJOINING,
                                                                             WORD_SCORE, STOP_WORDS[ind])))

    RESULT = True

    assert RESULT, 'Keywords are not extracted'
