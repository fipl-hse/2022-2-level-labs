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
                                              load_stop_words, get_top_n, generate_stop_words)


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


    def operations(text: str, stop_word: Sequence[str]) -> None:
        """
        Functions from main.py altogether
        """
        candidate_phrases, word_degree, word_score, freq_dict, cumulative_score, candidate_phrases_with_adjoining, \
            cumulative_score_for_candidates_with_stop_words, top_list = [None for i in range(8)]
        phrases = extract_phrases(text)
        if phrases and stop_word:
            candidate_phrases = extract_candidate_keyword_phrases(phrases, stop_word)

        if candidate_phrases:
            freq_dict = calculate_frequencies_for_content_words(candidate_phrases)

        if candidate_phrases and freq_dict:
            word_degree = calculate_word_degrees(candidate_phrases, list(freq_dict.keys()))

        if word_degree and freq_dict:
            word_score = calculate_word_scores(word_degree, freq_dict)

        if candidate_phrases and word_score:
            cumulative_score = calculate_cumulative_score_for_candidates(candidate_phrases, word_score)

        if cumulative_score:
            top_list = get_top_n(cumulative_score, 10, 5)
            print(top_list)

        if candidate_phrases and phrases:
            candidate_phrases_with_adjoining = extract_candidate_keyword_phrases_with_adjoining(candidate_phrases,
                                                                                                phrases)
        if candidate_phrases_with_adjoining and word_score and stop_word:
            cumulative_score_for_candidates_with_stop_words = \
                calculate_cumulative_score_for_candidates_with_stop_words(candidate_phrases_with_adjoining,
                                                                          word_score, stop_word)

        if cumulative_score_for_candidates_with_stop_words:
            top_with_stop_words = get_top_n(cumulative_score_for_candidates_with_stop_words, 10, 5)
            print(top_with_stop_words)

    for name, texts in corpus.items():
        print(str(name))
        operations(texts, stop_words)

    stop_words_dict = load_stop_words(TARGET_PATH_STOP_WORDS)

    if stop_words_dict:
        print('polish')
        operations(read_target_text(ASSETS_PATH / 'polish.txt'), stop_words_dict['pl'])

    generated_stop_words = generate_stop_words(read_target_text(TARGET_TEXT_PATH_UNKNOWN), 5)
    if generated_stop_words:
        print('unknown')
        operations(read_target_text(TARGET_TEXT_PATH_UNKNOWN), generated_stop_words)

        RESULT = 'the end'

        assert RESULT, 'Keywords are not extracted'
