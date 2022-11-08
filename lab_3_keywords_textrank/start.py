"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from main import TextPreprocessor, TextEncoder, extract_pairs, AdjacencyMatrixGraph, VanillaTextRank


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

    processed = TextPreprocessor(stop_words, punctuation)
    words = processed.preprocess_text(text)
    encoded = TextEncoder()
    tokens = encoded.encode(words)
    pairs = extract_pairs(tokens, 7)
    print(pairs)
    graph = AdjacencyMatrixGraph()
    for pair in pairs:
        graph.add_edge(pair[0], pair[1])
    van_graph = VanillaTextRank(graph)
    van_graph.train()
    top10 = van_graph.get_top_keywords(10)
    decoded10 = encoded.decode(top10)
    print(decoded10)





    #RESULT = decoded10
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    #assert RESULT, 'Keywords are not extracted'
