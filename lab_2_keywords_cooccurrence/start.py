"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from main import (extract_phrases,
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
    text = corpus['gagarin']

    if text:
        phrases = extract_phrases(text)

    if phrases and stop_words:
        candidate_keywords_phrases = extract_candidate_keyword_phrases(phrases, stop_words)

    if candidate_keywords_phrases:
        frequencies = calculate_frequencies_for_content_words(candidate_keywords_phrases)

    content_words = list(frequencies.keys())
    if candidate_keywords_phrases and content_words:
        word_degrees = calculate_word_degrees(candidate_keywords_phrases, content_words)

    if word_degrees and frequencies:
        word_scores = calculate_word_scores(word_degrees, frequencies)

    if candidate_keywords_phrases and word_scores:
        keyword_phrases_with_scores = calculate_cumulative_score_for_candidates(candidate_keywords_phrases, word_scores)

    TOP_N = 5
    MAX_LENGTH = 3
    if keyword_phrases_with_scores and TOP_N and MAX_LENGTH:
        print(get_top_n(keyword_phrases_with_scores, TOP_N, MAX_LENGTH))

    if candidate_keywords_phrases and phrases:
        final_phrases = extract_candidate_keyword_phrases_with_adjoining(candidate_keywords_phrases, phrases)

    if candidate_keywords_phrases and stop_words and word_scores:
        cumulative_score = calculate_cumulative_score_for_candidates_with_stop_words(candidate_keywords_phrases,
                                                                                     word_scores, stop_words)


    RESULT = cumulative_score

    assert RESULT, 'Keywords are not extracted'
