"""
Co-occurrence-driven keyword extraction starter
"""
from pathlib import Path
from typing import Optional
from lab_2_keywords_cooccurrence.main import (extract_phrases, extract_candidate_keyword_phrases,
                                              calculate_frequencies_for_content_words, calculate_word_degrees,
                                              calculate_word_scores, calculate_cumulative_score_for_candidates,
                                              get_top_n, extract_candidate_keyword_phrases_with_adjoining,
                                              calculate_cumulative_score_for_candidates_with_stop_words,
                                              generate_stop_words, load_stop_words)


def read_target_text(file_path: Path) -> str:
    """
    Utility functions that reads the text content from the file
    :param file_path: the path to the file
    :return: the text content of the file
    """
    with open(file_path, 'r', encoding='utf-8') as target_text_file:
        return target_text_file.read()


def key_phrases(text: str, stop_w: list) -> Optional[dict]:
    """
    checks functions
    """
    candidate_keyword_phrases = None
    freq = None
    word_degree = None
    word_score = None
    score_for_candidates = None
    candidate_phrases_with_adjoining = None

    extract_phrase = extract_phrases(text)

    if extract_phrase:
        print(candidate_keyword_phrases := extract_candidate_keyword_phrases(extract_phrase, stop_w))

    if candidate_keyword_phrases:
        freq = calculate_frequencies_for_content_words(candidate_keyword_phrases)

    if freq and candidate_keyword_phrases:
        word_degree = calculate_word_degrees(candidate_keyword_phrases, list(freq.keys()))

    if word_degree and freq:
        word_score = calculate_word_scores(word_degree, freq)

    if word_score and candidate_keyword_phrases:
        score_for_candidates = calculate_cumulative_score_for_candidates(candidate_keyword_phrases, word_score)

    if score_for_candidates:
        print(get_top_n(score_for_candidates, 10, 20))

    if candidate_keyword_phrases and extract_phrase:
        candidate_phrases_with_adjoining = extract_candidate_keyword_phrases_with_adjoining(
            candidate_keyword_phrases, extract_phrase)

    if candidate_phrases_with_adjoining and word_score:
        print(significant_phrases :=
              calculate_cumulative_score_for_candidates_with_stop_words(candidate_phrases_with_adjoining, word_score,
                                                                        stop_w))
        return dict(significant_phrases)
    return None


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

    for key in corpus:
        key_phrases(corpus[key], stop_words)

    # polish
    STOP_WORDS = load_stop_words(ASSETS_PATH / 'stopwords.json')
    if STOP_WORDS:
        key_phrases(read_target_text(ASSETS_PATH / 'polish.txt'), list(STOP_WORDS["pl"]))

    # unknown (Esperanto)
    TEXT_UNKNOWN = read_target_text(ASSETS_PATH / 'unknown.txt')
    STOP_UNKNOWN = generate_stop_words(TEXT_UNKNOWN, 20)
    if STOP_UNKNOWN:
        esperanto = key_phrases(TEXT_UNKNOWN, list(STOP_UNKNOWN))

        RESULT = esperanto
        assert RESULT, 'Keywords are not extracted'
