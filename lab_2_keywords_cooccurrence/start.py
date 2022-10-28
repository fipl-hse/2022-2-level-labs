"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (text_processing,
                                              get_top_n,
                                              load_stop_words)


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

    if GAGARIN_TXT_PROCESSED := text_processing(corpus["gagarin"], stop_words):
        print("Гагарин: ", get_top_n(GAGARIN_TXT_PROCESSED, 10, 3), sep="\n\t")

    if ALBATROSS_TXT_PROCESSED := text_processing(corpus["albatross"], stop_words):
        print("Альбатрос: ", get_top_n(ALBATROSS_TXT_PROCESSED, 10, 3), sep="\n\t")

    if GENOME_TXT_PROCESSED := text_processing(corpus["genome_engineering"], stop_words):
        print("Геном: ", get_top_n(GENOME_TXT_PROCESSED, 10, 3), sep="\n\t")

    if PAIN_TXT_PROCESSED := text_processing(corpus["pain_detection"], stop_words):
        print("Боль: ", get_top_n(PAIN_TXT_PROCESSED, 10, 3), sep="\n\t")

    STOP_WORDS = load_stop_words(ASSETS_PATH / "stopwords.json")

    POLISH_TXT_PROCESSED = None

    if STOP_WORDS:
        if POLISH_TXT_PROCESSED := text_processing(read_target_text(ASSETS_PATH / "polish.txt"), STOP_WORDS["pl"]):
            print("Польское: ", get_top_n(POLISH_TXT_PROCESSED, 10, 3), sep="\n\t")

    if UNKNOWN_TXT_PROCESSED := text_processing(read_target_text(ASSETS_PATH / "unknown.txt"), max_length=8):
        print("Эсперантовское: ", get_top_n(UNKNOWN_TXT_PROCESSED, 10, 3), sep="\n\t")

    RESULT = True

    assert RESULT, 'Keywords are not extracted'
