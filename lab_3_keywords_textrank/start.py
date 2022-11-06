"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation

from lab_3_keywords_textrank.main import TextPreprocessor, TextEncoder, extract_pairs


if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / 'article.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = tuple(file.read().split('\n'))

    preprocessor = TextPreprocessor(stop_words, tuple(punctuation))
    clean_tokens = preprocessor._clean_and_tokenize(text)
    print(clean_tokens)
    without_stop_words = preprocessor._remove_stop_words(clean_tokens)
    print(without_stop_words)
    prerocess_text = preprocessor.preprocess_text(text)
    print(prerocess_text)
    encoder = TextEncoder()
    ids_tokens = encoder.encode(prerocess_text)
    print(ids_tokens)
    token_ids = encoder.decode(ids_tokens)
    print(token_ids)
    pairs = extract_pairs(ids_tokens, 3)
    print(pairs)

    # RESULT = clean_tokens
    # # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    # assert RESULT, 'Keywords are not extracted'
