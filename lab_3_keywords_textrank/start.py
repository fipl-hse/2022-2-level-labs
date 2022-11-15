"""
TextRank keyword extraction starter
"""

from pathlib import Path
from string import punctuation
from lab_3_keywords_textrank.main import TextPreprocessor, TextEncoder, \
    AdjacencyMatrixGraph, EdgeListGraph, VanillaTextRank, PositionBiasedTextRank

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

    the_text = TextPreprocessor(stop_words, tuple(punct for punct in punctuation))
    tokens = the_text.preprocess_text(text)
    encoded_text = TextEncoder()
    encoded_tokens = encoded_text.encode(tokens)

    text_graph = AdjacencyMatrixGraph()
    text_graph.fill_from_tokens(encoded_tokens, 5)
    text_rank1 = VanillaTextRank(text_graph)
    text_rank1.train()
    top1 = text_rank1.get_top_keywords(10)
    decoded_text1 = encoded_text.decode(top1)
    print(decoded_text1)
    text_graph = EdgeListGraph()
    text_graph.fill_from_tokens(encoded_tokens, 5)
    text_rank1 = VanillaTextRank(text_graph)
    text_rank1.train()
    top1 = text_rank1.get_top_keywords(10)
    decoded_text1 = encoded_text.decode(top1)
    print(decoded_text1)

    text_graph = AdjacencyMatrixGraph()
    text_graph.fill_from_tokens(encoded_tokens, 2)
    text_graph.fill_positions(encoded_tokens)
    text_graph.calculate_position_weights()
    text_rank2 = PositionBiasedTextRank(text_graph)
    text_rank2.train()
    top2 = text_rank2.get_top_keywords(10)
    decoded_text2 = encoded_text.decode(top2)
    print(decoded_text2)
    text_graph = EdgeListGraph()
    text_graph.fill_from_tokens(encoded_tokens, 3)
    text_graph.fill_positions(encoded_tokens)
    text_graph.calculate_position_weights()
    text_rank2 = PositionBiasedTextRank(text_graph)
    text_rank2.train()
    top2 = text_rank2.get_top_keywords(10)
    decoded_text2 = encoded_text.decode(top2)
    print(decoded_text2)

    RESULT = decoded_text2
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
