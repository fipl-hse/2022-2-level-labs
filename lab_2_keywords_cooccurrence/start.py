"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (get_top_n,
                                              load_stop_words,
                                              extract_keyword_phrases,
                                              calculate_cumulative_score)


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

    STOP_WORDS_PATH = ASSETS_PATH / 'stopwords.json'
    POLISH_TEXT_PATH = ASSETS_PATH / 'polish.txt'
    UNKNOWN_TEXT_PATH = ASSETS_PATH / 'unknown.txt'

    gagarin_cumulative_score = calculate_cumulative_score(corpus['gagarin'], stop_words)
    if gagarin_cumulative_score:
        print(get_top_n(gagarin_cumulative_score, 5, 5))

    albatross_cumulative_score = calculate_cumulative_score(corpus['albatross'], stop_words)
    if albatross_cumulative_score:
        print(get_top_n(albatross_cumulative_score, 5, 5))

    genome_engineering_cumulative_score = calculate_cumulative_score(corpus['genome_engineering'], stop_words)
    if genome_engineering_cumulative_score:
        print(get_top_n(genome_engineering_cumulative_score, 5, 5))

    pain_detection_cumulative_score = calculate_cumulative_score(corpus['pain_detection'], stop_words)
    if pain_detection_cumulative_score:
        print(get_top_n(pain_detection_cumulative_score, 5, 5))

    stop_words = load_stop_words(STOP_WORDS_PATH)
    if stop_words:
        print(extract_keyword_phrases(read_target_text(POLISH_TEXT_PATH), stop_words['pl']))

    unknown_text = read_target_text(UNKNOWN_TEXT_PATH)
    if unknown_text:
        unknown_text_cumulative_score = calculate_cumulative_score(unknown_text)
        if unknown_text_cumulative_score:
            print(extract_keyword_phrases(unknown_text))
            print(unknown_text_cumulative_score)

    RESULT = 'Hello'
    assert RESULT, 'Keywords are not extracted'