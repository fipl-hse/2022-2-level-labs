"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import extract_phrases, extract_candidate_keyword_phrases, \
    calculate_frequencies_for_content_words, calculate_word_degrees,  \
    calculate_word_scores, calculate_cumulative_score_for_candidates, get_top_n, \
    extract_candidate_keyword_phrases_with_adjoining, \
    calculate_cumulative_score_for_candidates_with_stop_words


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
    for text in corpus.values():
        if text:
            phrases = extract_phrases(text)
        if phrases and stop_words:
            key_phrases = extract_candidate_keyword_phrases(phrases, stop_words)
        if key_phrases:
            word_frequencies = calculate_frequencies_for_content_words(key_phrases)
        if word_frequencies and key_phrases:
            word_degrees = calculate_word_degrees(key_phrases, list(word_frequencies.keys()))
        if word_degrees and word_frequencies:
            word_scores = calculate_word_scores(word_degrees, word_frequencies)
        if word_scores and key_phrases:
            cumulative_scores = calculate_cumulative_score_for_candidates(key_phrases, word_scores)
        if cumulative_scores:
            print(get_top_n(cumulative_scores, 5, 2))
        if key_phrases and phrases:
            key_phrases_with_sw = extract_candidate_keyword_phrases_with_adjoining(key_phrases, phrases)
        if key_phrases_with_sw and word_scores and stop_words:
            cumulative_scores_with_sw = calculate_cumulative_score_for_candidates_with_stop_words(key_phrases_with_sw,
                                                                                              word_scores, stop_words)
        if cumulative_scores_with_sw:
            print(get_top_n(cumulative_scores_with_sw, 5, 4), '\n')


    RESULT = cumulative_scores_with_sw

    assert RESULT, 'Keywords are not extracted'
