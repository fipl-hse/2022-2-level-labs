"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from main import extract_phrases, extract_candidate_keyword_phrases,\
    calculate_frequencies_for_content_words, calculate_word_degrees, \
    calculate_word_scores, calculate_cumulative_score_for_candidates, \
    get_top_n, extract_candidate_keyword_phrases_with_adjoining, \
    calculate_cumulative_score_for_candidates_with_stop_words
from lab_1_keywords_tfidf.main import clean_and_tokenize

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
    phrases = {}
    for text in corpus:
        phrases[text] = extract_phrases(corpus[text])

    key_pharases = {}
    for text in corpus:
        key_pharases[text] = extract_candidate_keyword_phrases(phrases[text], stop_words)
    for text, pharases in key_pharases.items():
        print(f'{text}:', *pharases)
        continue
    # print()

    frequencies = {}
    for text in corpus:
        frequencies[text] = calculate_frequencies_for_content_words(key_pharases[text])

    word_degrees = {}
    for text in corpus:
        word_degrees[text] = calculate_word_degrees(key_pharases[text], list(frequencies[text].keys()))

    word_scores = {}
    for text in corpus:
        word_scores[text] = calculate_word_scores(word_degrees[text], frequencies[text])

    cumulative_scores = {}
    for text in corpus:
        cumulative_scores[text] = calculate_cumulative_score_for_candidates(key_pharases[text], word_scores[text])

    for text in corpus:
        print(text, get_top_n(cumulative_scores[text], 5, 3))
        continue

    with_adjoining = {}
    for text in corpus:
        with_adjoining[text] = key_pharases[text] \
                               + extract_candidate_keyword_phrases_with_adjoining(key_pharases[text], phrases[text])

    candidates_with_stop_words = {}
    for text in corpus:
        candidates_with_stop_words[text] \
            = calculate_cumulative_score_for_candidates_with_stop_words(key_pharases[text], word_scores[text], stop_words)

    for text in corpus:
        print(text, get_top_n(candidates_with_stop_words[text], 5, 3))
        continue

    RESULT = corpus

    assert RESULT, 'Keywords are not extracted'
