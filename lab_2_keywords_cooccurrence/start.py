"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (extract_phrases, extract_candidate_keyword_phrases,
                                              calculate_frequencies_for_content_words, calculate_word_degrees,
                                              calculate_word_scores, calculate_cumulative_score_for_candidates,
                                              get_top_n, extract_candidate_keyword_phrases_with_adjoining)


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

    RESULT = None

    text = corpus['gagarin']

    if text:
        extracted_phrases = extract_phrases(text)
        print(extracted_phrases)

    if extracted_phrases and stop_words:
        candidate_extracted_phrases = extract_candidate_keyword_phrases(extracted_phrases, stop_words)
        print(candidate_extracted_phrases)

    if candidate_extracted_phrases:
        frequency_extracted_word = calculate_frequencies_for_content_words(candidate_extracted_phrases)
        print(frequency_extracted_word)

    if candidate_extracted_phrases and frequency_extracted_word:
        word_degree = calculate_word_degrees(candidate_extracted_phrases, list(frequency_extracted_word))
        print(word_degree)

    if word_degree and frequency_extracted_word:
        word_score = calculate_word_scores(word_degree, frequency_extracted_word)
        print(word_score)

    if word_score and candidate_extracted_phrases:
        extracted_phrases_score = calculate_cumulative_score_for_candidates(candidate_extracted_phrases, word_score)
        print(extracted_phrases_score)

    if extracted_phrases_score:
        top_phrases = get_top_n(extracted_phrases_score, 3, 5)
        print(top_phrases)

    print()
    RESULT = "Done"

    assert RESULT, 'Keywords are not extracted'
