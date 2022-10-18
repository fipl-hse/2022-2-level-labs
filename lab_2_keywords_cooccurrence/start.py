"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from typing import Sequence
from lab_2_keywords_cooccurrence.main import (extract_phrases,
                  extract_candidate_keyword_phrases,
                  calculate_frequencies_for_content_words,
                  calculate_word_degrees,
                  calculate_word_scores,
                  calculate_cumulative_score_for_candidates,
                  get_top_n,
                  extract_candidate_keyword_phrases_with_adjoining,
                  calculate_cumulative_score_for_candidates_with_stop_words,
                  load_stop_words,
                  generate_stop_words)

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


    def keyword_phrases(text: str, stop_w: Sequence[str]) -> None:

        phrases = extract_phrases(text)

        if phrases:
            candidate_keyword_phrases = extract_candidate_keyword_phrases(phrases, stop_w)

        if candidate_keyword_phrases:
            content_words = calculate_frequencies_for_content_words(candidate_keyword_phrases)

        content_words_list = list(content_words.keys())

        if candidate_keyword_phrases and content_words_list:
            word_degrees = calculate_word_degrees(candidate_keyword_phrases, content_words_list)

        if word_degrees and content_words:
            word_scores = calculate_word_scores(word_degrees, content_words)

        if word_scores and candidate_keyword_phrases:
            keyword_phrases_with_scores = \
                calculate_cumulative_score_for_candidates(candidate_keyword_phrases, word_scores)

        if keyword_phrases_with_scores:
            top_list = get_top_n(keyword_phrases_with_scores, 2, 3)
            print(top_list)

        if candidate_keyword_phrases:
            candidate_kw_phrases_w_adjoining = \
                extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases, phrases)

        if candidate_kw_phrases_w_adjoining and word_scores and stop_w:
            cumulative_score_with_stop_words = \
                calculate_cumulative_score_for_candidates_with_stop_words(candidate_kw_phrases_w_adjoining,
                                                                          word_scores, stop_w)
            print(cumulative_score_with_stop_words)


    keyword_phrases(corpus['gagarin'], stop_words)

    new_stop_words_dict = load_stop_words(ASSETS_PATH / 'stopwords.json')

    polish = read_target_text(ASSETS_PATH / 'polish.txt')
    if new_stop_words_dict and polish:
        keyword_phrases(polish, new_stop_words_dict['pl'])

    esperanto = read_target_text(ASSETS_PATH / 'unknown.txt')
    stop_words_esperanto = generate_stop_words(esperanto, 5)
    print(stop_words_esperanto)

    if esperanto and stop_words_esperanto:
        keyword_phrases(esperanto, stop_words_esperanto)

    RESULT = 'Esperanto'
    assert RESULT, 'Keywords are not extracted'
