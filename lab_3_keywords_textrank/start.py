"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import extract_pairs, TextEncoder, \
    TextPreprocessor, VanillaTextRank, AdjacencyMatrixGraph
    #, get_top_keywords


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

    processor = TextPreprocessor(punctuation, stop_words)
    encoder = TextEncoder()
    tokens =  encoder.encode(processor.preprocess_text(text))
    graph = AdjacencyMatrixGraph()
    graph.fill_from_tokens(tokens, 5)
    print(graph._matrix)

    ranking = VanillaTextRank(graph)
    ranking.train()
    print(extract_pairs(tokens, 3))
    print(encoder.decode(ranking.get_top_keywords(10)))
    RESULT = 'None'
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
