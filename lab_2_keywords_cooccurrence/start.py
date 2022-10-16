"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from typing import Sequence
from lab_2_keywords_cooccurrence.main import (extract_phrases, extract_candidate_keyword_phrases,
                                              calculate_frequencies_for_content_words, calculate_word_degrees,
                                              calculate_word_scores, calculate_cumulative_score_for_candidates,
                                              extract_candidate_keyword_phrases_with_adjoining,
                                              calculate_cumulative_score_for_candidates_with_stop_words,
                                              load_stop_words, get_top_n)


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
    TARGET_PATH_STOP_WORDS = ASSETS_PATH / 'stopwords.json'

    corpus = {
        'gagarin': read_target_text(TARGET_TEXT_PATH_GAGARIN),
        'albatross': read_target_text(TARGET_TEXT_PATH_ALBATROSS),
        'genome_engineering': read_target_text(TARGET_TEXT_PATH_GENOME),
        'pain_detection': read_target_text(TARGET_TEXT_PATH_PAIN_DETECTION)
    }

    # def extract_keyword_phrases(text: str, stop_words_list: Sequence[str]):
    candidate_phrases = None
    word_degree = None
    word_score = None
    freq_dict = None
    cumulative_score = None
    candidate_phrases_with_adjoining = None
    cumulative_score_for_candidates_with_stop_words = None

    TARGET_PATH_POLISH = read_target_text(ASSETS_PATH / 'polish.txt')
    TARGET_PATH_GAGARIN = read_target_text(TARGET_TEXT_PATH_GAGARIN)

    phrases = extract_phrases(TARGET_PATH_GAGARIN)
    if phrases and stop_words:
        candidate_phrases = extract_candidate_keyword_phrases(phrases, stop_words)
    if candidate_phrases:
        freq_dict = calculate_frequencies_for_content_words(candidate_phrases)
    if candidate_phrases and freq_dict:
        word_degree = calculate_word_degrees(candidate_phrases, list(freq_dict.keys()))
#       print(word_degree)
    if word_degree and freq_dict:
        word_score = calculate_word_scores(word_degree, freq_dict)
    # print(word_score)
    if candidate_phrases and word_score:
        cumulative_score = calculate_cumulative_score_for_candidates(candidate_phrases, word_score)
        print(cumulative_score)
    print(get_top_n(cumulative_score, 3, 5))
    if candidate_phrases and phrases:
        candidate_phrases_with_adjoining = extract_candidate_keyword_phrases_with_adjoining(candidate_phrases,
                                                                                            phrases)
    if candidate_phrases_with_adjoining and word_score and stop_words:
        cumulative_score_for_candidates_with_stop_words = \
            calculate_cumulative_score_for_candidates_with_stop_words(candidate_phrases_with_adjoining,
                                                                      word_score, stop_words)
    # return cumulative_score

    stop_words_dict = load_stop_words(TARGET_PATH_STOP_WORDS)
    # print(cumulative_score(TARGET_PATH_POLISH, stop_words_dict['pl']))
    # print(TARGET_PATH_GAGARIN, stop_words)

    # print(extract_keyword_phrases(TARGET_TEXT_PATH_POLISH, stop_words_dict['pl']))
    # polish_phrases =

    # RESULT = None
    #
    # assert RESULT, 'Keywords are not extracted'
