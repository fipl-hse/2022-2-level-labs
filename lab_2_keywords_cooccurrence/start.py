"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (extract_phrases, extract_candidate_keyword_phrases,
                                              calculate_frequencies_for_content_words, calculate_word_degrees,
                                              calculate_word_scores, calculate_cumulative_score_for_candidates,
                                              extract_candidate_keyword_phrases_with_adjoining)


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
    TARGET_TEXT_PATH_ALBATROSS = ASSETS_PATH / 'albatross.txt'
    TARGET_TEXT_PATH_PAIN_DETECTION = ASSETS_PATH / 'pain_detection.txt'
    TARGET_TEXT_PATH_GAGARIN = ASSETS_PATH / 'gagarin.txt'
    TARGET_TEXT_PATH_GENOME = ASSETS_PATH / 'genome_engineering.txt'

    corpus = {
        'gagarin': read_target_text(TARGET_TEXT_PATH_GAGARIN),
        'albatross': read_target_text(TARGET_TEXT_PATH_ALBATROSS),
        'genome_engineering': read_target_text(TARGET_TEXT_PATH_GENOME),
        'pain_detection': read_target_text(TARGET_TEXT_PATH_PAIN_DETECTION)
    }

    # target_text = read_target_text(TARGET_TEXT_PATH_GAGARIN)

    phrases = extract_phrases(read_target_text(TARGET_TEXT_PATH_GAGARIN))
    print(phrases)
    candidate_phrases = extract_candidate_keyword_phrases(phrases, stop_words)
    # print(candidate_phrases)
    freq_dict = calculate_frequencies_for_content_words(candidate_phrases)
    # print(freq_dict)
    word_degree = calculate_word_degrees(candidate_phrases, list(freq_dict.keys()))
    # print(word_degree)
    word_score = calculate_word_scores(word_degree, freq_dict)
    # print(word_score)
    cumulative_score = calculate_cumulative_score_for_candidates(candidate_phrases, word_score)
    # print(main.get_top_n(cumulative_score, 3, 5))
    candidate_phrases_with_adjoining = extract_candidate_keyword_phrases_with_adjoining(candidate_phrases, phrases)
    print(candidate_phrases_with_adjoining)

    # RESULT = None
    #
    # assert RESULT, 'Keywords are not extracted'
