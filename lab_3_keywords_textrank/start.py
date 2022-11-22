"""
TextRank keyword extraction starter
"""

from pathlib import Path
from lab_3_keywords_textrank.main import (TextPreprocessor, TextEncoder, extract_pairs,
                                          AdjacencyMatrixGraph, VanillaTextRank)


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

    prep = TextPreprocessor(stop_words, ('!', ',', '(', ')', ':', '/', '-', '.'))
    tokens = prep.preprocess_text(text)
    print(tokens)

    encoder = TextEncoder()
    encoded = encoder.encode(tokens)
    # print(encoded)
    decoded = encoder.decode(encoded)
    # print(decoded)

    if encoded:
        pairs = extract_pairs(encoded, 3)

    graph = AdjacencyMatrixGraph()
    graph.fill_from_tokens(encoded, 3)

    #
    # print(graph.add_edge(1001, 1005))
    # print(graph.is_incidental(1001, 1005))
    # print(graph.calculate_inout_score(1001))
    vanilla_text_rank = VanillaTextRank(graph)
    # if encoded:
    vanilla_text_rank.train()
    top_10 = vanilla_text_rank.get_top_keywords(10)
    top = encoder.decode(top_10)
    # print(top)

    RESULT = top
    # # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
