"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from typing import Optional, Sequence

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


def analysis(text: str, stops: list) -> Optional[Sequence[str]]:
    new_top = []
    phrases = extract_phrases(text)

    if phrases:
        candidates = extract_candidate_keyword_phrases(phrases, stops)

    if candidates:
        frequencies_for_content_words = calculate_frequencies_for_content_words(candidates)

    if candidates and frequencies_for_content_words:
        word_degrees = calculate_word_degrees(candidates, list(frequencies_for_content_words.keys()))

    if word_degrees and frequencies_for_content_words:
        word_scores = calculate_word_scores(word_degrees, frequencies_for_content_words)

    if candidates and word_scores:
        cumulative_score_for_candidates = calculate_cumulative_score_for_candidates(candidates, word_scores)

    #if cumulative_score_for_candidates:
        #top = get_top_n(cumulative_score_for_candidates, 10, 3)
        #print(top)

    if candidates and phrases:
        candidates_with_adjoining = extract_candidate_keyword_phrases_with_adjoining(candidates, phrases)

    if stop_words and candidates_with_adjoining and word_scores:
        cumulative_score_for_candidates_wsw = calculate_cumulative_score_for_candidates_with_stop_words(
            candidates_with_adjoining, word_scores, stops)

    if cumulative_score_for_candidates_wsw and cumulative_score_for_candidates:
        merged_cum = {**cumulative_score_for_candidates, **cumulative_score_for_candidates_wsw}
        new_top = get_top_n(merged_cum, 10, 2)
        #print(new_top)

    return new_top


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


    print(analysis(corpus['gagarin'], stop_words))

    dict_of_stop_words = load_stop_words(ASSETS_PATH / 'stopwords.json')
    print(dict_of_stop_words)

    polish = read_target_text(ASSETS_PATH / 'polish.txt')
    if dict_of_stop_words and polish:
        polish_text_analysed = analysis(polish, dict_of_stop_words['pl'])
        print(polish_text_analysed)

    esperanto_text = read_target_text(ASSETS_PATH / 'unknown.txt')
    stops_for_esperanto = generate_stop_words(esperanto_text, 5)
    print(stops_for_esperanto)

    if esperanto_text and stops_for_esperanto:
        esperanto_results = analysis(esperanto_text, stops_for_esperanto)
        print(esperanto_results)

    RESULT = esperanto_results

    assert RESULT, 'Keywords are not extracted'
