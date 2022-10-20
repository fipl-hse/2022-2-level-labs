"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from main import (extract_phrases, extract_candidate_keyword_phrases, calculate_frequencies_for_content_words,
                  calculate_word_degrees, calculate_word_scores, calculate_cumulative_score_for_candidates,
                  get_top_n, extract_candidate_keyword_phrases_with_adjoining,
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

    RESULT = None
    gagarin = corpus['gagarin']
    top_n = 2
    max_length = 3
    extracted_phrases = extract_phrases(gagarin)
    candidate_keyword_phrases = extract_candidate_keyword_phrases(extracted_phrases, stop_words)
    frequencies_for_content_words = calculate_frequencies_for_content_words(candidate_keyword_phrases)
    word_degrees = calculate_word_degrees(candidate_keyword_phrases, list(frequencies_for_content_words.keys()))
    word_scores = calculate_word_scores(word_degrees, frequencies_for_content_words)
    cumulative_score_for_candidates = calculate_cumulative_score_for_candidates(candidate_keyword_phrases, word_scores)
    top_n_phrases = get_top_n(cumulative_score_for_candidates, top_n, max_length)
    #candidate_keyword_phrases_with_adjoining = extract_candidate_keyword_phrases_with_adjoining(
    #    candidate_keyword_phrases, extracted_phrases)
    candidate_keyword_phrases_with_adjoining = extract_candidate_keyword_phrases_with_adjoining([('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'), ('времена', 'союза', 'прошли'),
                                     ('одной',), ('важнейших', 'задач'), ('исследование', 'космоса'),
                                     ('ящик',), ('железа',), ('самое',), ('ящик',), ('железа',)], ['Во времена Советского Союза исследование космоса было одной из важнейших задач',
                   'И именно из СССР была отправлена ракета',
                   'совершившая первый полет в космос',
                   'Произошло это 12 апреля 1961 года',
                   'было это одной из важнейших задач',
                   'Ящик для железа не то же самое, что ящик из железа '])
    cumulative_score_for_candidates_with_stop_words = calculate_cumulative_score_for_candidates_with_stop_words(
        candidate_keyword_phrases_with_adjoining, word_scores, stop_words)

    print('extracted_phrases = ', extracted_phrases, '\n',
          'candidate_keyword_phrases = ',  candidate_keyword_phrases, '\n',
          'frequencies_for_content_words = ', frequencies_for_content_words, '\n',
          'word_degrees = ',  word_degrees, '\n',
          'word_scores = ',  word_scores, '\n',
          'cumulative_score_for_candidates = ',  cumulative_score_for_candidates, '\n',
          'top_n_phrases = ', top_n_phrases, '\n',
          'candidate_keyword_phrases_with_adjoining = ', candidate_keyword_phrases_with_adjoining, '\n',
          'cumulative_score_for_candidates_with_stop_words = ', cumulative_score_for_candidates_with_stop_words)

    RESULT = top_n_phrases

    assert RESULT, 'Keywords are not extracted'
