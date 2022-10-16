"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from main import (extract_phrases, extract_candidate_keyword_phrases, calculate_frequencies_for_content_words,
                  calculate_word_degrees, calculate_word_scores, calculate_cumulative_score_for_candidates,
                  get_top_n, extract_candidate_keyword_phrases_with_adjoining,
                  calculate_cumulative_score_for_candidates_with_stop_words, load_stop_words)


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

    NUM_FOR_TOP = 0
    for key, values in corpus.items():
        (EXTRACTED_PHRASES, CANDIDATE_KEY_PHR, FREQ_CONT, CONTENT_WORDS, WORD_DEGREE, WORD_SCORE,
         CUMULATIVE_SCORE, KEY_PHR_ADJOIN, CUMULATIVE_SCORE_ADJOIN) = [None for notdef in range(9)]
        FILE_NAME = key
        FILE_READ = values
        NUM_FOR_TOP += 2
        if FILE_READ:
            EXTRACTED_PHRASES = extract_phrases(FILE_READ)
        if EXTRACTED_PHRASES:
            CANDIDATE_KEY_PHR = extract_candidate_keyword_phrases(EXTRACTED_PHRASES, stop_words)

        if CANDIDATE_KEY_PHR:
            FREQ_CONT = calculate_frequencies_for_content_words(CANDIDATE_KEY_PHR)

        CONTENT_WORDS = [keys for keys in FREQ_CONT.keys()]  # is this how we're supposed to extract content words?

        if CANDIDATE_KEY_PHR:
            WORD_DEGREE = calculate_word_degrees(CANDIDATE_KEY_PHR, CONTENT_WORDS)

        if WORD_DEGREE and FREQ_CONT:
            WORD_SCORE = calculate_word_scores(WORD_DEGREE, FREQ_CONT)

        if WORD_SCORE and CANDIDATE_KEY_PHR:
            CUMULATIVE_SCORE = calculate_cumulative_score_for_candidates(CANDIDATE_KEY_PHR, WORD_SCORE)

        if CUMULATIVE_SCORE:
            TOP_N = get_top_n(CUMULATIVE_SCORE, NUM_FOR_TOP, NUM_FOR_TOP + 1)
            print('Top key phrases without stop words: ', TOP_N)

        if CANDIDATE_KEY_PHR and EXTRACTED_PHRASES:
            KEY_PHR_ADJOIN = extract_candidate_keyword_phrases_with_adjoining(CANDIDATE_KEY_PHR, EXTRACTED_PHRASES)

        if KEY_PHR_ADJOIN and WORD_SCORE:
            CUMULATIVE_SCORE_ADJOIN = calculate_cumulative_score_for_candidates_with_stop_words(KEY_PHR_ADJOIN,
                                                                                                WORD_SCORE, stop_words)

        if CUMULATIVE_SCORE_ADJOIN:
            TOP_N_ADJOIN = get_top_n(CUMULATIVE_SCORE_ADJOIN, NUM_FOR_TOP, NUM_FOR_TOP + 1)
            print('Top key phrases with adjoining: ', TOP_N_ADJOIN)

    (EXTRACTED_PHRASES_POLISH, CANDIDATE_KEY_PHR_POLISH, FREQ_CONT_POLISH, CONTENT_WORDS_POLISH, WORD_DEGREE_POLISH,
     WORD_SCORE_POLISH, KEY_PHR_ADJOIN_POLISH) = [None for notdef in range(9)]

    JSON_PATH = ASSETS_PATH / 'stopwords.json'
    POLISH_STOPS = load_stop_words(JSON_PATH)
    POLISH_FILE = ASSETS_PATH / 'polish.txt'
    POLISH_TEXT = read_target_text(POLISH_FILE)

    if POLISH_TEXT:
        EXTRACTED_PHRASES_POLISH = extract_phrases(POLISH_TEXT)

    if EXTRACTED_PHRASES_POLISH:
        CANDIDATE_KEY_PHR_POLISH = extract_candidate_keyword_phrases(EXTRACTED_PHRASES_POLISH, POLISH_STOPS)

    if CANDIDATE_KEY_PHR_POLISH:
        FREQ_CONT_POLISH = calculate_frequencies_for_content_words(CANDIDATE_KEY_PHR_POLISH)

    CONTENT_WORDS_POLISH = [keys for keys in FREQ_CONT_POLISH.keys()]

    if CANDIDATE_KEY_PHR_POLISH:
        WORD_DEGREE_POLISH = calculate_word_degrees(CANDIDATE_KEY_PHR_POLISH, CONTENT_WORDS_POLISH)

    if WORD_DEGREE_POLISH and FREQ_CONT_POLISH:
        WORD_SCORE_POLISH = calculate_word_scores(WORD_DEGREE_POLISH, FREQ_CONT_POLISH)

    if CANDIDATE_KEY_PHR_POLISH and EXTRACTED_PHRASES_POLISH:
        KEY_PHR_ADJOIN_POLISH = extract_candidate_keyword_phrases_with_adjoining(CANDIDATE_KEY_PHR_POLISH,
                                                                                 EXTRACTED_PHRASES_POLISH)

    print(KEY_PHR_ADJOIN_POLISH)  # do we need to print it?

    # UNKNOWN_PATH = ASSETS_PATH / 'unknown.txt'
    # UNKNOWN_TXT = read_target_text(UNKNOWN_PATH)
    # this task is not finished yet

    RESULT = KEY_PHR_ADJOIN_POLISH
    assert RESULT, 'Keywords are not extracted'
