"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from typing import Optional
from lab_2_keywords_cooccurrence.main import extract_phrases, extract_candidate_keyword_phrases, \
    calculate_frequencies_for_content_words, calculate_word_degrees, calculate_word_scores, \
    calculate_cumulative_score_for_candidates, get_top_n, extract_candidate_keyword_phrases_with_adjoining, \
    calculate_cumulative_score_for_candidates_with_stop_words, load_stop_words, generate_stop_words


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

    def keyphrases(text: str, stopwords: list) -> Optional[dict]:
        phrases = extract_phrases(text)

        if phrases and stopwords:
            candidate_keyword_phrases = extract_candidate_keyword_phrases(phrases, stopwords)

        if candidate_keyword_phrases:
            content_words = calculate_frequencies_for_content_words(candidate_keyword_phrases)

        if content_words and candidate_keyword_phrases:
            word_degrees = calculate_word_degrees(candidate_keyword_phrases, list(content_words.keys()))

        if word_degrees and content_words:
            word_scores = calculate_word_scores(word_degrees, content_words)

        if word_scores and candidate_keyword_phrases:
            keyword_phrases_with_scores = calculate_cumulative_score_for_candidates(candidate_keyword_phrases,
                                                                                    word_scores)

        if keyword_phrases_with_scores:
            print(get_top_n(keyword_phrases_with_scores, 10, 5))

        if candidate_keyword_phrases and phrases:
            candidate_keyphrases_adjoining = extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases,
                                                                                              phrases)

        if candidate_keyphrases_adjoining and word_scores and stopwords:
            cumulative_score_with_stopwords = calculate_cumulative_score_for_candidates_with_stop_words(
                candidate_keyphrases_adjoining, word_scores, stopwords)

            if cumulative_score_with_stopwords:
                return dict(cumulative_score_with_stopwords)
        return None

    keyphrases(corpus['gagarin'], stop_words)
    keyphrases(corpus['genome_engineering'], stop_words)
    keyphrases(corpus['albatross'], stop_words)
    keyphrases(corpus['pain_detection'], stop_words)

    polish_stopwords = load_stop_words(ASSETS_PATH/'stopwords.json')
    if polish_stopwords:
        print(keyphrases(read_target_text(ASSETS_PATH/'polish.txt'), list(polish_stopwords['pl'])))

    unknown_stopwords = generate_stop_words(read_target_text(ASSETS_PATH/'unknown.txt'), 5)
    if unknown_stopwords:
        unknown_keyphrases = keyphrases(read_target_text(ASSETS_PATH/'unknown.txt'), list(unknown_stopwords))
        print(unknown_keyphrases)

    RESULT = unknown_keyphrases

    assert RESULT, 'Keywords are not extracted'
