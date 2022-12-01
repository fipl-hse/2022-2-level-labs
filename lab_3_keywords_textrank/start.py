"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import extract_pairs, TextPreprocessor, TextEncoder, AdjacencyMatrixGraph, \
    VanillaTextRank


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

    PREPROCESSOR = TextPreprocessor(stop_words, tuple(punctuation))
    TOKENS = PREPROCESSOR.preprocess_text(text)

    ENCODER = TextEncoder()
    ENCODED_TOKENS = ENCODER.encode(TOKENS)
    if ENCODED_TOKENS:
        print(extract_pairs(ENCODED_TOKENS, 3))

    adjacency_matrix_graph = AdjacencyMatrixGraph()
    if ENCODED_TOKENS:
        adjacency_matrix_graph.fill_from_tokens(ENCODED_TOKENS, 3)
    vanilla_text_rank_amg = VanillaTextRank(adjacency_matrix_graph)
    vanilla_text_rank_amg.train()
    print(ENCODED_TOKENS.decode(vanilla_text_rank_amg.get_top_keywords(10)))

    RESULT = True

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'