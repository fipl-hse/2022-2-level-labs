"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path

from main import extract_phrases, extract_candidate_keyword_phrases, calculate_frequencies_for_content_words, \
    calculate_word_degrees, calculate_word_scores, get_top_n, \
    extract_candidate_keyword_phrases_with_adjoining, calculate_cumulative_score_for_candidates_with_stop_words


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

    RESULT = {}

    for title, text in corpus.items():
        extracted_phrases = extract_phrases(text)
        candidate_keywords = extract_candidate_keyword_phrases(extracted_phrases, stop_words)
        ajoined_candidate_keywords = extract_candidate_keyword_phrases_with_adjoining(candidate_keywords,
                                                                                      extracted_phrases)
        if ajoined_candidate_keywords:
            candidate_keywords += ajoined_candidate_keywords
        word_frequencies = calculate_frequencies_for_content_words(candidate_keywords)
        word_degrees = calculate_word_degrees(candidate_keywords, list(word_frequencies.keys()))
        word_scores = calculate_word_scores(word_degrees, word_frequencies)
        cumulative_score_with_stopwords = calculate_cumulative_score_for_candidates_with_stop_words(
            candidate_keywords, word_scores, stop_words)
        top_n = get_top_n(cumulative_score_with_stopwords, 10, 5)
        RESULT[title] = top_n
        print(title, top_n)

    assert RESULT, 'Keywords are not extracted'
