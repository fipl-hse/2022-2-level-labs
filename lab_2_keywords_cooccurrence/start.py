"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (
    process_text,
    get_top_n,
    load_stop_words
)

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

    GAGARIN_PROCESSED = process_text(corpus['gagarin'], stop_words)
    if GAGARIN_PROCESSED:
        print(get_top_n(GAGARIN_PROCESSED, 10, 5))
    ALBATROSS_PROCESSED = process_text(corpus['albatross'], stop_words)
    if ALBATROSS_PROCESSED:
        print(get_top_n(ALBATROSS_PROCESSED, 10, 5))
    GENOME_PROCESSED = process_text(corpus['genome_engineering'], stop_words)
    if GENOME_PROCESSED:
        print(get_top_n(GENOME_PROCESSED, 10, 5))
    PAIN_PROCESSED = process_text(corpus['pain_detection'], stop_words)
    if PAIN_PROCESSED:
        print(get_top_n(PAIN_PROCESSED, 10, 5))

    STOP_WORDS = load_stop_words(ASSETS_PATH / 'stopwords.json')

    POLISH_PROCESSED = None
    if STOP_WORDS:
        POLISH_PROCESSED = process_text(read_target_text(ASSETS_PATH / 'polish.txt'), STOP_WORDS['pl'])
    if POLISH_PROCESSED:
        print(get_top_n(POLISH_PROCESSED, 10, 5))

    UNKNOWN_PROCESSED = process_text(read_target_text(ASSETS_PATH / 'unknown.txt'), max_length=8)
    if UNKNOWN_PROCESSED:
        print(get_top_n(UNKNOWN_PROCESSED, 10, 5))  # эсперанто

    RESULT = UNKNOWN_PROCESSED

    assert RESULT, 'Keywords are not extracted'
