"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from typing import Sequence

from lab_2_keywords_cooccurrence.main import (
    extract_phrases,
    extract_candidate_keyword_phrases,
    calculate_frequencies_for_content_words,
    calculate_word_degrees,
    calculate_word_scores,
    calculate_cumulative_score_for_candidates,
    get_top_n,
    extract_candidate_keyword_phrases_with_adjoining,
    calculate_cumulative_score_for_candidates_with_stop_words,
    generate_stop_words,
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

    TARGET_PATH_STOP_WORDS = ASSETS_PATH / 'stopwords.json'
    TARGET_PATH_POLISH = ASSETS_PATH / 'polish.txt'
    TARGET_TEXT_PATH_UNKNOWN = ASSETS_PATH / 'unknown.txt'

    corpus = {
        'gagarin': read_target_text(TARGET_TEXT_PATH_GAGARIN),
        'albatross': read_target_text(TARGET_TEXT_PATH_ALBATROSS),
        'genome_engineering': read_target_text(TARGET_TEXT_PATH_GENOME),
        'pain_detection': read_target_text(TARGET_TEXT_PATH_PAIN_DETECTION)
    }

    def functions(text: str, stop_words: Sequence[str]) -> None:
        phrases = extract_phrases(text)

        if phrases and stop_words:
            candidate_keyword_phrases = extract_candidate_keyword_phrases(phrases, stop_words)

        if candidate_keyword_phrases:
            word_frequency = calculate_frequencies_for_content_words(candidate_keyword_phrases)

        if candidate_keyword_phrases and word_frequency:
            word_degrees = calculate_word_degrees(candidate_keyword_phrases, list(word_frequency.keys()))

        if word_degrees and word_frequency:
            word_scores = calculate_word_scores(word_degrees, word_frequency)

        if candidate_keyword_phrases and word_scores:
            keyword_phrases_with_scores = calculate_cumulative_score_for_candidates(candidate_keyword_phrases,
                                                                                    word_scores)

        if keyword_phrases_with_scores:
            top_lst = get_top_n(keyword_phrases_with_scores, 10, 6)
            print(f'top n lst: {top_lst}')

        if candidate_keyword_phrases and phrases:
            candidate_keyword_phrases_with_adj = extract_candidate_keyword_phrases_with_adjoining(
                candidate_keyword_phrases, phrases)
            print(f'keyword phrases with stop words: {candidate_keyword_phrases_with_adj}')

        if candidate_keyword_phrases and word_scores and stop_words:
            keyword_phrases_with_scores_with_stops = calculate_cumulative_score_for_candidates_with_stop_words(
                candidate_keyword_phrases, word_scores, stop_words)

        if keyword_phrases_with_scores_with_stops:
            top_lst_with_stops = get_top_n(keyword_phrases_with_scores_with_stops, 10, 6)
            print(f'top n lst with stop words: {top_lst_with_stops}')

        print()


    for title, story in corpus.items():
        print(f'info about the text called {title}')
        functions(story, stop_words)

    polish_text = read_target_text(TARGET_PATH_POLISH)
    diff_lang_stop_words = load_stop_words(ASSETS_PATH / 'stopwords.json')
    if polish_text and diff_lang_stop_words:
        functions(polish_text, diff_lang_stop_words['pl'])

    unknown_text = read_target_text(TARGET_TEXT_PATH_UNKNOWN)
    unknown_stop_words = generate_stop_words(unknown_text, 10)
    if unknown_text and unknown_stop_words:
        functions(unknown_text, unknown_stop_words)


    RESULT = 'finished'

    assert RESULT, 'Keywords are not extracted'
