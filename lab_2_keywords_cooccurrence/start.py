"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (
    process_text,
    get_top_n,
    load_stop_words
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

    for text in corpus:
        processed_text = process_text(text, stop_words)
        if processed_text:
            print('text', get_top_n(processed_text, 10, 4), "\n")

    polish_text = read_target_text(ASSETS_PATH / 'polish.txt')
    stop_words = load_stop_words(ASSETS_PATH / 'stopwords.json')  # Incompatible types in assignment
    # (expression has type "Optional[Mapping[str, Sequence[str]]]", variable has type "List[str]")  [assignment]
    if stop_words:
        processed_polish = process_text(polish_text, stop_words['pl'])
        if processed_polish:
            print('polish_text', get_top_n(processed_polish, 10, 4), "\n")  # Incompatible types in assignment
        # (expression has type "Optional[Mapping[str, Sequence[str]]]", variable has type "List[str]")  [assignment]

    unknown_text = read_target_text(ASSETS_PATH / 'unknown.txt')
    processed_unknown = process_text(unknown_text, stop_words)
    if processed_unknown:
        print('unknown_text', get_top_n(processed_unknown, 10, 4), "\n")  # Argument 1 to "get_top_n" has
        # incompatible type "Optional[Mapping[Tuple[str, ...], float]]";
        # expected "Mapping[Tuple[str, ...], float]"  [arg-type]

RESULT = True

assert RESULT, 'Keywords are not extracted'
